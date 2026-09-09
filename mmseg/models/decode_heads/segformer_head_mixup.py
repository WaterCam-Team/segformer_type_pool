# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr
# from mmseg.models.decode_heads.coop import load_clip_to_cpu, CustomCLIP
from mmseg.models.decode_heads.coop_my import CustomPrompt
from IPython import embed
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class CrossAttention(nn.Module):
    """Cross Attention module for text-visual feature fusion"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Add text projection layer to match dimensions
        self.text_proj = nn.Linear(512, dim)
        
        # For visual features (query)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # For text features (key, value) - now using projected dimension
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, text_feat):
        B, N, C = x.shape
        
        # Project text features to match visual dimension and expand batch size
        text_feat = self.text_proj(text_feat)  # [1, dim]
        text_feat = text_feat.expand(B, -1)  # Expand to [B, dim]
        
        # Project text features to key and value
        k = self.k(text_feat).unsqueeze(1)  # [B, 1, C]
        v = self.v(text_feat).unsqueeze(1)  # [B, 1, C]
        
        # Project visual features to query
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

@HEADS.register_module()
class SegFormerHead_Mixup(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, use_mixup=False,mixup_alpha=0.4,**kwargs):
        super(SegFormerHead_Mixup, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )        
            
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        # self.text_proj = nn.Linear(512, 256)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)


    def masked_pooled_alignment_loss(self, Fv, text_proj_emb, gt_mask, eps=1e-6):
        """
        Fv: (B, D, H, W) normalized
        text_proj_emb: (D,) or (B, D) normalized
        gt_mask: (B, H, W) float {0,1}
        returns scalar MSE loss on pooled cosine similarity
        """
        B, D, H, W = Fv.shape
        device = Fv.device

        # prepare text emb
        if text_proj_emb.dim() == 1:
            text = text_proj_emb.view(1, D).expand(B, -1)
        else:
            text = text_proj_emb
            
        gt_mask = F.interpolate(gt_mask.float(), size=Fv.shape[2:], mode="nearest").long()    
        gt_mask = (gt_mask > 0).to(torch.long)
        mask = gt_mask.to(Fv.dtype)  # (B,1,H,W)
        area = mask.sum(dim=(2,3))                 # (B,1)
        area_safe = torch.where(area==0, torch.ones_like(area), area)

        v_pool = (Fv * mask).sum(dim=(2,3)) / (area_safe + eps)   # (B, D)
        # fallback to global avg if no positive pixels
        no_pos = (area.squeeze(-1) == 0)
        if no_pos.any():
            v_global = Fv.view(B, D, -1).mean(-1)
            v_pool[no_pos] = v_global[no_pos]

        v_pool = v_pool / (v_pool.norm(dim=1, keepdim=True) + eps)
        text = text / (text.norm(dim=1, keepdim=True) + eps)

        sims = (v_pool * text).sum(dim=1)  # (B,)
        target = torch.ones_like(sims, device=device)
        # optional weighting by area ratio
        area_ratio = (area.squeeze(-1) / (H*W)).clamp(min=0.05)
        loss = ((sims - target)**2 * area_ratio).mean()
        return loss

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        # First apply MLP to get embeddings
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # Resize all features to the same size
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        # Fuse all features
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        
        # if self.use_mixup
        
        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x

class RepRTAHead(nn.Module):
    """
    Training-only lightweight visual->text projection.
    Input: feature map _c (B, C, H, W)
    Output: embedding map Fv (B, D, H, W) normalized
    """
    def __init__(self, in_ch, clip_dim=512, mid_ch=256):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(mid_ch)
        self.act  = nn.GELU()
        # project to CLIP dim
        self.proj = nn.Conv2d(mid_ch, clip_dim, kernel_size=1, bias=True)
        # init proj small
        nn.init.normal_(self.proj.weight, std=0.01)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.proj(x)                # (B, D, H, W)
        # normalize per-vector
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-6)
        return x_norm  
    
class TextProjector(nn.Module):
    def __init__(self, clip_dim=512, hidden_dim=2048, out_dim=512, activation=nn.ReLU()):
        """
        Two linear layers applied to CLIP text embedding (like YOLOE).
        hidden_dim can be large (paper often uses wide hidden).
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim, bias=True),
            activation,
            nn.Linear(hidden_dim, out_dim, bias=True)
        )
        # initialize small
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, text_emb):
        # text_emb: (D,) or (B,D)
        out = self.mlp(text_emb)
        out = out / (out.norm(dim=1, keepdim=True) + 1e-6) if out.dim() == 2 else out / (out.norm() + 1e-6)
        return out  # normalized