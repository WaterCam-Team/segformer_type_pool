import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
import open_clip
import clip
import json
CHALLENGING_TYPES = ['transparent', 'shallow', 'reflection', 'glare', 'dark', 'muddy', 'rainy', 'blurry']

@SEGMENTORS.register_module()
class EncoderDecoder_denseclip_attribute(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 classnames,
                 K_water,
                 K_bg,
                 context_length,
                 neck=None,
                 auxiliary_head=None,
                 identity_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoder_denseclip_attribute, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        
        self.with_identity_head = False
        self.identity_head = None
        self._init_identity_head(identity_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)
        ### coop prompt learner ###
        self.device = "cuda"
        # self.K_per_cls = [K_bg, K_water]
        self.classnames = classnames
        # self.context_length = context_length
        # self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
        # self.clip_model = self.clip_model.float()
        # self.clip_model.eval()
        # clip_model, _ = clip.load("ViT-B/16", device=self.device)
        # clip_model = clip_model.float()
        # clip_model.eval()
        # for p in clip_model.parameters():
        #     p.requires_grad = False
        # self.half_prompt_learner =  Half_CoOpPromptLearner_type_pool(
        #     classnames=self.classnames,
        #     clip_model=clip_model,
        #     n_ctx=8).to(self.device)

        # # self.prompt_learner = CoOpPromptLearner(
        # #     self.classnames,
        # #     clip_model,
        # #     self.K_per_cls,
        # #     n_ctx=self.context_length).to(self.device)
        # # self.K_per_cls.append(K_water)
        # # self.prompt_learner = CoOpPromptLearner_flood_water(
        # #     self.classnames,
        # #     clip_model,
        # #     self.K_per_cls,
        # #     n_ctx=self.context_length).to(self.device)
        # self.text_encoder = TextEncoder(clip_model)
        ### difficulty prompt learner#####
        self.device = "cuda"
        self.clip_model, self.clip_tokenizer = self.build_clip()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        # self._clip_model_ref = [self.clip_model]  # list is not registered by nn.Module
        # del self.clip_model
        # self.prompt_learner = Difficulty_PromptLearner(
        #     class_names=classnames)
        self.half_prompt_learner = Half_PromptLearner_type_tool(
            class_names=classnames,
            clip_model=self.clip_model,
            clip_tokenizer=self.clip_tokenizer)
        
        ### adding context encoder ###
        # self.context_decoder = ContextDecoder()
        self.gamma = nn.Parameter(torch.ones(512) * 1e-4)
        # self.context_decoder = TransformerDecoderLayer(512, 8, 0.1)
        self.context_decoder = nn.ModuleList([
                    TransformerDecoderLayer(512, 8, 0.1) for _ in range(3)
                ])
        ###############################
        ### fixed text embeddings for water and non-water ###
        # water_text_feat_path = "/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat/mean_water_feat.pt"
        # non_water_text_feat_path = "/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat/non_water_feat.pt"
        # clip_water_text_embeddings = torch.load(water_text_feat_path, map_location='cuda',weights_only=True)
        # clip_non_water_text_embeddings = torch.load(non_water_text_feat_path, map_location='cuda',weights_only=True)
        # clip_text_embeddings = torch.stack([clip_non_water_text_embeddings, clip_water_text_embeddings], dim=0)
        # self.register_buffer("E_WATER", clip_text_embeddings)
        # fixed_prompt_path  = "/data1/huantao/workspace/project/flood_seg/dataset_w_urban_flood/water_scene_descriptions.json"
        # with open(fixed_prompt_path, "r") as f:
        #     self.fixed_prompt_dict = json.load(f)
        ###################################
        
        self.tau = 0.07
        self.visual_proj = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=1,
            bias=False)
        
        assert self.with_decode_head

    def build_clip(self):
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        clip_model = clip_model.cuda().eval()
        return clip_model, tokenizer
    
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)
    
    def _init_identity_head(self, identity_head):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder_denseclip_attribute, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def after_extract_feat(self, x, fixed_prompts, difficulty):
        x_orig = list(x[0:4])
        n = x[3].shape[0]
        pixel_feat = self.visual_proj(x[3])                              # [B,512,H,W]
        ### coop prompt learner ###
        # prompts, tokenized_prompts = self.half_prompt_learner(pixel_feat)
        # text_feat_ori = self.text_encoder(prompts, tokenized_prompts)
        ##### prompt learner ######
        text_feat_ori = self.half_prompt_learner(pixel_feat)
        ###### adding context encoding ######
        B, C, H, W = pixel_feat.shape
        visual_context = torch.cat([F.adaptive_avg_pool2d(pixel_feat, (1, 1)).squeeze(-1), pixel_feat.reshape(B, C, H*W)],dim=2).permute(0, 2, 1)
        fg_text = text_feat_ori[:B]          # (bs, C)  — one per image
        bg_text = text_feat_ori[B:]          # (1, C)   — shared background
        bg_text = bg_text.expand(B, -1)   # (bs, C)
        text_feat = torch.stack([bg_text, fg_text], dim=1)
        # text_diff = self.context_decoder(text_feat, visual_context)
        text_diff = text_feat
        for layer in self.context_decoder:
            text_diff = layer(text_diff, visual_context)
        text_feat = text_feat + self.gamma * text_diff
        ##### select based on difficulty type ######
        # bg_feat = text_feat[:,0,:]
        # if len(difficulty)!=0:
        #     water_feats = []
        #     for b in range(B):
        #         water_feat =  text_feat[b, difficulty[b]+1] 
        #         water_feats.append(water_feat)
        #     water_feats = torch.stack(water_feats, dim=0)     
        # else:
        #     water_feats = torch.mean(text_feat[:,1:,:],dim=1)
        # text_feat = torch.stack([bg_feat, water_feats], dim=1)           
        ##############################
        pixel_feat = pixel_feat / (pixel_feat.norm(dim=1, keepdim=True) + 1e-6)
        # text_feat = text_feat.unsqueeze(0).expand(n, -1, -1)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        score_map = torch.einsum('bchw,bkc->bkhw', pixel_feat, text_feat)
        x_orig[3] = torch.cat([x_orig[3], score_map], dim=1)
        return x_orig, score_map
    
    def after_extract_feat_water_flood(self, x, img_metas):
        x_orig = list(x[0:4])
        n = x[3].shape[0]
        pixel_feat = self.visual_proj(x[3])                              # [B,512,H,W]
        ### coop prompt learner ###
        prompts, tokenized_prompts = self.prompt_learner()
        text_feat_all = self.text_encoder(prompts, tokenized_prompts)
        ### random select one per class ###
        K_bg, K_water, K_flood = self.K_per_cls
        bg_feats = text_feat_all[:K_bg]                    # (K_bg, C)
        water_feats = text_feat_all[K_bg:K_bg+K_water]    # (K_water, C)
        flood_feats = text_feat_all[K_bg+K_water:] 
        #### Assign prompt based on image type
        selected_text_feats = []
        for i, meta in enumerate(img_metas):
            filename = meta.get("filename", "")
            if filename.endswith("_flood.jpg"):
                # Randomly select one prompt from flood class
                idx_flood = torch.randint(0, K_flood, (1,), device=text_feat_all.device)
                water_feat_i = flood_feats[idx_flood]
            else:
                # Randomly select one prompt from water class
                idx_water = torch.randint(0, K_water, (1,), device=text_feat_all.device)
                water_feat_i = water_feats[idx_water]

            # Always select one prompt from background
            idx_bg = torch.randint(0, K_bg, (1,), device=text_feat_all.device)
            bg_feat_i = bg_feats[idx_bg]

            # Concatenate background + water/flood feature for this image
            text_feat_i = torch.cat([bg_feat_i, water_feat_i], dim=0)  # (2, C)
            selected_text_feats.append(text_feat_i)
        text_feat = torch.stack(selected_text_feats, dim=0)  # (B, 2, C)
        ##############################
        pixel_feat = pixel_feat / (pixel_feat.norm(dim=1, keepdim=True) + 1e-6)
        # text_feat = text_feat.unsqueeze(0).expand(n, -1, -1)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        ### fixed text embeddings ###
        # text_feat = self.E_WATER.unsqueeze(0).expand(n, -1, -1)
        # text_feat = text_feat / (text_feat.norm() + 1e-6)       # [512]
        ##############################
        score_map = torch.einsum('bchw,bkc->bkhw', pixel_feat, text_feat)
        x_orig[3] = torch.cat([x_orig[3], score_map], dim=1)
        
        return x_orig, score_map
    
    def encode_text_list(self, fixed_prompts):
        bg_prompts = []
        water_prompts = []
        for text in fixed_prompts:
            # split into two sentences
            sentences = text.strip().split(". ")
            if len(sentences) >= 2:
                res1 = sentences[0]
                res2 = sentences[-1]
            else:
                res1 = text
                res2 = text

            if not res1.endswith("."):
                res1 += "."
            bg_prompts.append(res1)
            water_prompts.append(res2)
        # tokenize
        bg_tokens = clip.tokenize(bg_prompts).to(self.device)
        water_tokens = clip.tokenize(water_prompts).to(self.device)

        # encode
        with torch.no_grad():
            bg_features = self.clip_model.encode_text(bg_tokens)
            water_features = self.clip_model.encode_text(water_tokens)
        text_features = torch.stack([bg_features, water_features], dim=1)

        return text_features

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        x, score_map = self.after_extract_feat(x, [], [])
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses
    
    def _identity_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        loss_aux = self.identity_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_aux, 'aux_identity'))
        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)
        # #### hard code the text embeddings for water and non-water ####
        # water_text_feat_path = "/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat/mean_water_feat.pt"
        # # # water_text_feat_path = '/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat/mean_water_flood_feat.pt'
        # non_water_text_feat_path = "/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat/non_water_feat.pt"
        # clip_water_text_embeddings = torch.load(water_text_feat_path, map_location='cuda',weights_only=True)
        # clip_non_water_text_embeddings = torch.load(non_water_text_feat_path, map_location='cuda',weights_only=True)
        # clip_text_embeddings = torch.stack([clip_non_water_text_embeddings, clip_water_text_embeddings], dim=0)
        # pixel_feat = self.visual_proj(x[-1])   # B × 1024 × H × W
        # pixel_feat = F.normalize(pixel_feat, dim=1)
        # text_tokens = self.to_mask_space(clip_text_embeddings)  # (C*K, hidden_dim)
        # text_tokens = text_tokens.unsqueeze(1).repeat(1,B, 1)       # (B, T, hidden_dim)
        # T = text_tokens.shape[0]
        #### 0: normal 1: transparent 2:shallow 3:reflection 4:glare 5:dark 6:muddy 7:rainy 8:blurry
        difficulty = []
        fixed_prompts = []
        for img_meta in img_metas:
            img_name = img_meta['filename'].split('/')[-1].split('_')
            if 'challenging' in img_name:
                challenging_type = img_name[-1].split('.')[0]
                type_idx = CHALLENGING_TYPES.index(challenging_type)+1
                difficulty.append(type_idx) 
            else:
                difficulty.append(0)
            # img_name_full = img_meta['filename'].split('/')[-1]
            # fixed_prompts.append(self.fixed_prompt_dict[img_name_full])
        difficulty = torch.tensor(difficulty, device=self.device)
        #########################
        x, score_map = self.after_extract_feat(x, fixed_prompts, difficulty)
        # x, score_map_list = self.after_extract_feat(x)
        # x, score_map = self.after_extract_feat_water_flood(x, img_metas)
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)
        
        if self.with_identity_head:
            loss_identity = self._identity_head_forward_train(
                score_map/self.tau, img_metas, gt_semantic_seg)
            losses.update(loss_identity)
            # loss_identity_total = 0.0
            # for i, score_map in enumerate(score_map_list):
            #     loss_identity_i = self._identity_head_forward_train(
            #         score_map/self.tau, img_metas, gt_semantic_seg)
            #     loss_identity_total += loss_identity_i['aux_identity.loss_seg']
            # losses['loss_aux_identity'] = loss_identity_total

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


class Half_CoOpPromptLearner_type_pool(nn.Module):
    def __init__(
        self,
        classnames,
        clip_model,
        n_ctx=8,
        ctx_init=None,
        device="cuda",
    ):
        super().__init__()
        
        self.classnames = classnames
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.device = device

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # self.clip_model = clip_model
        object.__setattr__(self, 'clip_model', clip_model)
        self.max_len = 77
        
        self.types = ['normal'] + CHALLENGING_TYPES
        self.types_with_water = [t + ' water' for t in self.types]
        type_tokens = clip.tokenize(self.types_with_water).to(device)
        with torch.no_grad():
            self.embedding_type = clip_model.encode_text(type_tokens).type(dtype)
        
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            tokens = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokens).type(dtype)
            init_ctx = embedding[0, 1:1+n_ctx, :]
            ctx = init_ctx.unsqueeze(0).repeat(K, 1, 1)
        else:
            # ctx = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            ctx = torch.empty(self.n_cls, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx, std=0.02)

            self.ctx = nn.Parameter(ctx)
        
    def forward(self, pixel_feat):
        pixel_global = pixel_feat.mean(dim=[2, 3]) 
        pixel_feat = F.normalize(pixel_global, dim=1)
        #### select one #####
        sim = torch.einsum('bc,nc->bn', pixel_global, self.embedding_type)
        bs = sim.shape[0]
        best_sim, best_idx = sim.max(dim=1)
        fg_prompts = []
        for b in range(bs):
            idx = best_idx[b].item()
            full_prompt = ' '.join(['X'] * self.n_ctx) + " " + self.types_with_water[idx]
            fg_prompts.append(full_prompt)

        # bg_prompts = ["The background of water"]
        ### bg is not fixed ###
        bg_prompts = [' '.join(['X'] * self.n_ctx) + ", the background of water"]
        ###########
        fg_tokenized = clip.tokenize(fg_prompts).to(self.device)
        bg_tokenized = clip.tokenize(bg_prompts).to(self.device)

        with torch.no_grad():
            fg_embedding = self.clip_model.token_embedding(fg_tokenized).type(self.clip_model.dtype)
            bg_embedding = self.clip_model.token_embedding(bg_tokenized).type(self.clip_model.dtype)

        # --- FG: replace placeholder tokens with learnable ctx ---
        fg_prefix = fg_embedding[:, :1, :]           # (bs, 1, dim)  SOS
        fg_suffix = fg_embedding[:, 1+self.n_ctx:, :] # (bs, L, dim)  class tokens + EOS
        # ctx = self.ctx.unsqueeze(0).expand(bs, -1, -1) # (bs, n_ctx, dim)
        ctx_fg = self.ctx[0].unsqueeze(0).expand(bs, -1, -1) 
        # fg_prompts_emb = torch.cat([fg_prefix, ctx, fg_suffix], dim=1)  # (bs, 77, dim)
        fg_prompts_emb = torch.cat([fg_prefix, ctx_fg, fg_suffix], dim=1)

        # --- BG: use the fixed embedding as-is, no ctx injection ---
        # bg_prompts_emb = bg_embedding  # (1, 77, dim)
        bg_prefix = bg_embedding[:, :1, :]              # (1, 1, dim)
        bg_suffix = bg_embedding[:, 1+self.n_ctx:, :]   # (1, L, dim)
        ctx_bg = self.ctx[1].unsqueeze(0)   
        bg_prompts_emb = torch.cat([bg_prefix, ctx_bg, bg_suffix], dim=1)           

        # --- Combine: [fg_0, fg_1, ..., fg_bs-1, bg] ---
        prompts = torch.cat([fg_prompts_emb, bg_prompts_emb], dim=0)  # (bs+1, 77, dim)
        tokenized = torch.cat([fg_tokenized, bg_tokenized], dim=0)     # (bs+1, 77)

        return prompts, tokenized

class Half_PromptLearner_type_tool(nn.Module):
    def __init__(
        self,
        class_names,
        context_len: int = 8,
        clip_model=None,
        clip_tokenizer=None,
        device='cuda'
    ):
        super().__init__()

        self.class_names = class_names
        self.types = ['normal'] + CHALLENGING_TYPES
        # self.C = num_classes
        self.context_len = context_len
        self.device = device

        assert clip_model is not None
        assert clip_tokenizer is not None

        object.__setattr__(self, 'clip_model', clip_model)
        object.__setattr__(self, 'clip_tokenizer', clip_tokenizer)

        clip_ctx_dim = clip_model.token_embedding.weight.shape[1]
        self.clip_ctx_dim = clip_ctx_dim
        self.clip_out_dim = clip_model.text_projection.shape[1]

        # index_to_replace = self.types.index("dark")
        # self.types[index_to_replace] = "nighttime" 
        # prefix replaced by context tokens
        self.full_class_names = []
        for cls in class_names:
            if cls == "water":
                self.full_class_names.extend([f"{t} water" for t in self.types])
            else:
                self.full_class_names.append(cls)
        # index_to_replace = self.full_class_names.index("dark water")
        # self.full_class_names[index_to_replace] = 'water in dark scene'
        tokens = self.clip_tokenizer(self.full_class_names[1:]).cuda()  
        with torch.no_grad():
            embedding_type = self.clip_model.encode_text(tokens)
        
        self.C = len(self.full_class_names)
        # learnable context tokens (C, context_len, D)
        self.prompt_ctx = nn.Parameter(torch.randn(self.C, context_len, clip_ctx_dim) * 0.02)
        self.prefix = " ".join(["X"] * context_len) + " "

        texts = [self.prefix + name for name in self.full_class_names]
        token_ids = self.clip_tokenizer(texts).to(self.device)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(token_ids)

        self.register_buffer("token_ids", token_ids)
        self.register_buffer("classname_emb", embedding)
        self.register_buffer("water_type_emb", embedding_type)

    def forward(self, pixel_feat):
        # pixel_feat = F.normalize(pixel_feat, dim=1)
        pixel_global = pixel_feat.mean(dim=[2, 3]) 
        pixel_global = F.normalize(pixel_global, dim=-1)
        #### select one #####
        sim = torch.einsum('bc,nc->bn', pixel_global, self.water_type_emb)
        # weights = F.softmax(sim / 0.07, dim=-1)
        # selected = torch.einsum('bn,nd->bd', weights, self.water_type_emb)
        best_sim, psudo_label = sim.max(dim=1)
        
        token_ids = self.token_ids
        classname_emb = self.classname_emb
        B_cls, L, D = classname_emb.shape
        outputs = []

        for ci in range(self.C):
            ctx = self.prompt_ctx[ci]  # (context_len, D)
            cls_emb = classname_emb[ci].clone()
            cls_emb[1:1+self.context_len] = ctx
            x = cls_emb + self.clip_model.positional_embedding
            x = x.unsqueeze(1)
            x = self.clip_model.transformer(x)
            x = x.squeeze(1)
            x = self.clip_model.ln_final(x)

            eos_idx = token_ids[ci].argmax().item()
            context_tokens = x[1:1+self.context_len]
            context_pooled = context_tokens.mean(dim=0)

            cls_tokens = x[1+self.context_len:eos_idx]
            cls_pooled = cls_tokens.mean(dim=0)

            text_feat = 0.7 * context_pooled + 0.3 * cls_pooled
            text_feat = text_feat @ self.clip_model.text_projection

            outputs.append(text_feat)
        outputs = torch.stack(outputs, dim=0)
        
        fg_indices = psudo_label + 1          # (bs,) — offset for background
        fg_text = outputs[fg_indices]          # (bs, clip_out_dim)
        bg_text = outputs[0].unsqueeze(0)
        # Stack: (bs, 2, clip_out_dim) — [bg, fg] per image
        selected_text_feat = torch.cat([fg_text, bg_text], dim=0) 
        return selected_text_feat
    
class Difficulty_PromptLearner(nn.Module):
    def __init__(
        self,
        class_names,
        context_len: int = 8,
        device='cuda'
    ):
        super().__init__()

        self.class_names = class_names
        self.types = ['normal'] + CHALLENGING_TYPES
        # self.C = num_classes
        self.context_len = context_len
        self.device = device
        
        clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
        clip_model = clip_model.cuda().eval()

        assert clip_model is not None
        assert clip_tokenizer is not None

        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        for p in self.clip_model.parameters():
            p.requires_grad = False

        clip_ctx_dim = clip_model.token_embedding.weight.shape[1]
        self.clip_ctx_dim = clip_ctx_dim
        self.clip_out_dim = clip_model.text_projection.shape[1]

        # index_to_replace = self.types.index("dark")
        # self.types[index_to_replace] = "nighttime" 
        # prefix replaced by context tokens
        self.full_class_names = []
        for cls in class_names:
            if cls == "water":
                self.full_class_names.extend([f"{t} water" for t in self.types])
            else:
                self.full_class_names.append(cls)
        # index_to_replace = self.full_class_names.index("dark water")
        # self.full_class_names[index_to_replace] = 'water in dark scene'
        
        self.C = len(self.full_class_names)
        # learnable context tokens (C, context_len, D)
        self.prompt_ctx = nn.Parameter(torch.randn(self.C, context_len, clip_ctx_dim) * 0.02)
        
        self.prefix = " ".join(["X"] * context_len) + " "

        texts = [self.prefix + name for name in self.full_class_names]
        token_ids = self.clip_tokenizer(texts).to(self.device)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(token_ids)

        self.register_buffer("token_ids", token_ids)
        self.register_buffer("classname_emb", embedding)

    def forward(self):
        token_ids = self.token_ids
        classname_emb = self.classname_emb
        B_cls, L, D = classname_emb.shape
        outputs = []
        for ci in range(self.C):

            ctx = self.prompt_ctx[ci]  # (context_len, D)

            cls_emb = classname_emb[ci].clone()

            cls_emb[1:1+self.context_len] = ctx

            x = cls_emb + self.clip_model.positional_embedding
            x = x.unsqueeze(1)
            x = self.clip_model.transformer(x)
            x = x.squeeze(1)
            x = self.clip_model.ln_final(x)

            eos_idx = token_ids[ci].argmax().item()

            context_tokens = x[1:1+self.context_len]
            context_pooled = context_tokens.mean(dim=0)

            cls_tokens = x[1+self.context_len:eos_idx]
            cls_pooled = cls_tokens.mean(dim=0)
            text_feat = 0.5 * context_pooled + 0.5 * cls_pooled
            text_feat = text_feat @ self.clip_model.text_projection
            outputs.append(text_feat)

        outputs = torch.stack(outputs, dim=0)

        return outputs
    


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        # EOS token (last non-zero)
        eos_idx = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_idx] @ self.text_projection

        return x
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x
