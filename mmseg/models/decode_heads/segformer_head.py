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
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, use_text=False,use_visual=False,**kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.use_text = use_text
        self.use_visual = use_visual

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        self.embed_dim = embedding_dim
        
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

        # Add FiLM layers with proper initialization
        # if self.use_text and self.use_yoloe:
        #     self.reprta_head = RepRTAHead(in_ch=self.embed_dim, clip_dim=512, mid_ch=256)
        #     self.text_proj = TextProjector(clip_dim=512, hidden_dim=2048, out_dim=512)
        #     # optionally store E_WATER as a buffer (CLIP text embedding)
        #     e_water = torch.load("/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat/mean_water_feat.pt")   # tensor[512], float32 cpu
        #     self.register_buffer("E_WATER", e_water) 
        if self.use_text:
            self.proj_to_clip = nn.Conv2d(embedding_dim, 512, kernel_size=1, bias=False)
            self.logit_scale = nn.Parameter(torch.tensor(10.0))      # learnable temperature
            self.bg_head     = nn.Conv2d(embedding_dim, 1, kernel_size=1)

            # load once (precomputed from your prompt ensemble)
            e_water = torch.load("/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat/mean_water_feat.pt")   # tensor[512], float32 cpu
            self.register_buffer("E_WATER", e_water)  # stays frozen
        elif self.use_visual:
            self.bg_head = nn.Conv2d(self.embed_dim, 1, 1)        # background branch
            self.logit_scale = nn.Parameter(torch.tensor(10.0))   # temperature for cosine logits
            self.register_buffer('p_water', torch.zeros(self.embed_dim))  # EMA visual prototype
            self.ema_m = 0.99                                     # EMA momentum
            self.water_id = 1 
                
            # self.film_mul_c4 = nn.Linear(512, c4_in_channels)
            # self.film_add_c4 = nn.Linear(512, c4_in_channels)
            # self.film_mul_c3 = nn.Linear(512, c3_in_channels)
            # self.film_add_c3 = nn.Linear(512, c3_in_channels)
            # self.film_mul_c2 = nn.Linear(512, c2_in_channels)
            # self.film_add_c2 = nn.Linear(512, c2_in_channels)
            # self.film_mul_c1 = nn.Linear(512, c1_in_channels)
            # self.film_add_c1 = nn.Linear(512, c1_in_channels)
            
            # self.cross_attn_c4 = CrossAttention(embedding_dim, num_heads=8)
            # self.cross_attn_c3 = CrossAttention(embedding_dim, num_heads=8)
            # self.cross_attn_c2 = CrossAttention(embedding_dim, num_heads=8)
            # self.cross_attn_c1 = CrossAttention(embedding_dim, num_heads=8)
            
            # Add layer normalization for each stage
            # self.norm_c4 = nn.LayerNorm(c4_in_channels)
            # self.norm_c3 = nn.LayerNorm(c3_in_channels)
            # self.norm_c2 = nn.LayerNorm(c2_in_channels)
            # self.norm_c1 = nn.LayerNorm(c1_in_channels)
            # self.norm_c4 = nn.LayerNorm(embedding_dim)
            # self.norm_c3 = nn.LayerNorm(embedding_dim)
            # self.norm_c2 = nn.LayerNorm(embedding_dim)
            # self.norm_c1 = nn.LayerNorm(embedding_dim)
            
            # Add learnable temperature parameter
            # self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        # Initialize FiLM layers
            # for m in [self.film_mul_c4, self.film_add_c4, 
            #          self.film_mul_c3, self.film_add_c3,
            #          self.film_mul_c2, self.film_add_c2,
            #          self.film_mul_c1, self.film_add_c1]:
            #     nn.init.xavier_uniform_(m.weight)
            #     nn.init.zeros_(m.bias)
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        # self.text_proj = nn.Linear(512, 256)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        
        # if self.use_coop:
        #     import clip
        #     device = "cuda" if torch.cuda.is_available() else "cpu"
        #     clip_model, preprocess = clip.load("ViT-B/16", device=device)
        #     self.prompt_learner = CustomPrompt(clip_model, class_name='water', position='end')

            

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape
        
        # if self.use_text:
        #     if self.use_coop:
        #        text_feat = self.prompt_learner().type(torch.float32)
        #     text_feat = torch.load('/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat//water_text_feat.pt').to(c1.dtype)
        #     c4 = c4.reshape(-1,n,self.in_channels[3])
        #     c4 = self.film_mul_c4(text_feat) * c4 + self.film_add_c4(text_feat)
        #     c4 = c4.reshape(n,-1,h4,w4)
            
        #     c3 = c3.reshape(-1,n,self.in_channels[2])
        #     c3 = self.film_mul_c3(text_feat) * c3 + self.film_add_c3(text_feat)
        #     c3 = c3.reshape(n,-1,h3,w3)
            
        #     c2 = c2.reshape(-1,n,self.in_channels[1])
        #     c2 = self.film_mul_c2(text_feat) * c2 + self.film_add_c2(text_feat)
        #     c2 = c2.reshape(n,-1,h2,w2)
            
        #     c1 = c1.reshape(-1,n,self.in_channels[0])
        #     c1 = self.film_mul_c1(text_feat) * c1 + self.film_add_c1(text_feat)
        #     c1 = c1.reshape(n,-1,h1,w1)

        # First apply MLP to get embeddings
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # Apply cross attention if using text features
        # if self.use_text:
        #     if self.use_coop:
        #         text_feat = self.prompt_learner().type(torch.float32)
        #     else:
        #         text_feat = torch.load('/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat/water_text_feat.pt').to(_c1.dtype)
            
        #     # Normalize text features
        #     text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        #     # Apply FiLM modulation with layer norm and temperature
        #     # Stage 4
        #     # c4_flat = c4.reshape(-1, n, self.in_channels[3])
        #     # c4_flat = self.norm_c4(c4_flat)
        #     # c4_flat = self.film_mul_c4(text_feat) * c4_flat + self.film_add_c4(text_feat)
        #     # c4_flat = c4_flat * self.temperature
        #     # c4 = c4_flat.reshape(n, -1, h4, w4)
        #     # Reshape features for attention
        #     _c4_flat = _c4.flatten(2).transpose(1, 2)  # [B, H*W, C]
        #     _c3_flat = _c3.flatten(2).transpose(1, 2)
        #     _c2_flat = _c2.flatten(2).transpose(1, 2)
        #     _c1_flat = _c1.flatten(2).transpose(1, 2)
            
        #     # Stage 3
        #     # c3_flat = c3.reshape(-1, n, self.in_channels[2])
        #     # c3_flat = self.norm_c3(c3_flat)
        #     # c3_flat = self.film_mul_c3(text_feat) * c3_flat + self.film_add_c3(text_feat)
        #     # c3_flat = c3_flat * self.temperature
        #     # c3 = c3_flat.reshape(n, -1, h3, w3)
        #     # Apply cross attention with layer norm
        #     _c4_flat = self.norm_c4(_c4_flat)
        #     _c4_flat = self.cross_attn_c4(_c4_flat, text_feat) * self.temperature
        #     _c4 = _c4_flat.transpose(1, 2).reshape(n, -1, h4, w4)
            
        #      # Stage 2
        #     # c2_flat = c2.reshape(-1, n, self.in_channels[1])
        #     # c2_flat = self.norm_c2(c2_flat)
        #     # c2_flat = self.film_mul_c2(text_feat) * c2_flat + self.film_add_c2(text_feat)
        #     # c2_flat = c2_flat * self.temperature
        #     # c2 = c2_flat.reshape(n, -1, h2, w2)
        #     _c3_flat = self.norm_c3(_c3_flat)
        #     _c3_flat = self.cross_attn_c3(_c3_flat, text_feat) * self.temperature
        #     _c3 = _c3_flat.transpose(1, 2).reshape(n, -1, h3, w3)
            
        #     _c2_flat = self.norm_c2(_c2_flat)
        #     _c2_flat = self.cross_attn_c2(_c2_flat, text_feat) * self.temperature
        #     _c2 = _c2_flat.transpose(1, 2).reshape(n, -1, h2, w2)
            
        #     _c1_flat = self.norm_c1(_c1_flat)
        #     _c1_flat = self.cross_attn_c1(_c1_flat, text_feat) * self.temperature
        #     _c1 = _c1_flat.transpose(1, 2).reshape(n, -1, h1, w1)

        # Resize all features to the same size
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        # Fuse all features
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        
        # if self.use_text:
        #     if self.use_coop:
        #        text_feat = self.prompt_learner().type(torch.float32)
        #     else:
        #         text_feat = torch.load('/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat/text_feat_512_combine.pt').to(_c.dtype)
        #         # text_feat = text_feat.mean(dim=0,keepdim=True)
        #         text_feat = text_feat[14].unsqueeze(0)
        #     text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        #     #### FiLM ###
        #     # scale = self.scale_fc(text_feat).unsqueeze(-1).unsqueeze(-1)
        #     # shift = self.shift_fc(text_feat).unsqueeze(-1).unsqueeze(-1)
        #     # _c = _c * scale + shift
        #     # _c = self.smooth(_c)
        #     #################
            
            ### add ###
            # bs, _, h, w = _c.shape
            # text_feat = self.text_proj(text_feat)
            # text_feat = text_feat.unsqueeze(-1).unsqueeze(-1)
            # text_feat = text_feat.expand(bs, -1, h, w) 
            # #_c = torch.cat([_c,text_feat], dim=1)
            # _c = _c + text_feat
        
        # if self.use_text and self.use_yoloe:
        #     Fv = self.reprta_head(_c)   
        #     return Fv       
        if self.use_text:
            F = self.proj_to_clip(_c)                              # [B,512,H,W]
            F = F / (F.norm(dim=1, keepdim=True) + 1e-6)

            Ew = self.E_WATER / (self.E_WATER.norm() + 1e-6)       # [512]
            logit_water = self.logit_scale * (F * Ew.view(1,-1,1,1)).sum(1, keepdim=True)  # [B,1,H,W]
            logit_bg    = self.bg_head(_c)                         # [B,1,H,W]
            x = torch.cat([logit_bg, logit_water], dim=1)
        elif self.use_visual:
            Fv = _c / ( _c.norm(dim=1, keepdim=True) + 1e-6 )          # [B,C,H,W]

            # Safe prototype normalization (handles zero at start)
            p = self.p_water
            p = p / (p.norm() + 1e-6)                                  # [C]
            # Water logit = cosine with visual prototype
            logit_water = (Fv * p.view(1, -1, 1, 1)).sum(1, keepdim=True)   # [B,1,H,W]
            logit_water = self.logit_scale.clamp(5., 20.) * logit_water
            # Background logit (learned conv)
            logit_bg = self.bg_head(_c)                                     # [B,1,H,W]
            x = torch.cat([logit_bg, logit_water], dim=1)                   # [B,2,H,W]
            return (x,_c)
        else:
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