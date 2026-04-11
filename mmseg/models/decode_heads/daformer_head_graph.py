# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
from .segformer_head import MLP
from .sep_aspp_head import DepthwiseSeparableASPPModule

from mmseg.ops import resize
import numpy as np
import torch.nn.functional as F

from .euler_margin import Euler_Attention

class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d=256):
        super(Affinity, self).__init__()
        self.d = d

        self.fc_M = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)

        )
        self.project_sr = nn.Linear(256, 256,bias=False)
        self.project_tg = nn.Linear(256, 256,bias=False)
        self.reset_parameters()
    def reset_parameters(self):
        for i in self.fc_M:
            if isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, std=0.01)
                nn.init.constant_(i.bias, 0)
        nn.init.normal_(self.project_sr.weight, std=0.01)
        nn.init.normal_(self.project_tg.weight, std=0.01)
    def forward(self, X, Y):

        X = self.project_sr(X)
        Y = self.project_tg(Y)

        N1, C = X.size()
        N2, C = Y.size()

        X_k = X.unsqueeze(1).expand(N1, N2, C)
        Y_k = Y.unsqueeze(0).expand(N1, N2, C)
        M = torch.cat([X_k, Y_k], dim=-1)
        M = self.fc_M(M).squeeze()
        return M

class dot_attention(nn.Module):

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale        
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)     
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention
class MultiHeadAttention_Graph(nn.Module):
    def __init__(self, model_dim=256, num_heads=4, dropout=0.0, version='v2'):
        super(MultiHeadAttention_Graph, self).__init__()

        self.dim_per_head = model_dim//num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.version  = version

    def forward(self, key, value, query, attn_mask=None):

        if self.version == 'v2':
            B =1
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            query = query.unsqueeze(1)
            residual = query
            dim_per_head = self.dim_per_head
            num_heads = self.num_heads
            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)

            key = key.view(key.size(0), B * num_heads, dim_per_head).transpose(0,1)
            value = value.view(value.size(0), B * num_heads, dim_per_head).transpose(0,1)
            query = query.view(query.size(0), B * num_heads, dim_per_head).transpose(0,1)

            scale = (key.size(-1) // num_heads) ** -0.5
            context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
            # (query, key, value, scale, attn_mask)
            context = context.transpose(0, 1).contiguous().view(query.size(1), B, dim_per_head * num_heads)
            output = self.linear_final(context)
            # dropout
            output = self.dropout(output)
            output = self.layer_norm(residual + output)
            # output = residual + output

        elif self.version == 'v1': # some difference about the place of torch.view fuction
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            query = query.unsqueeze(0)
            residual = query
            B, L, C = key.size()
            dim_per_head = self.dim_per_head
            num_heads = self.num_heads
            batch_size = key.size(0)

            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)

            key = key.view(batch_size * num_heads, -1, dim_per_head)
            value = value.view(batch_size * num_heads, -1, dim_per_head)
            query = query.view(batch_size * num_heads, -1, dim_per_head)

            if attn_mask:
                attn_mask = attn_mask.repeat(num_heads, 1, 1)
            scale = (key.size(-1) // num_heads) ** -0.5
            context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
            context = context.view(batch_size, -1, dim_per_head * num_heads)
            output = self.linear_final(context)
            output = self.dropout(output)
            output = self.layer_norm(residual + output)

        return output.squeeze(), attention.squeeze()

class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    elif type == 'isa':
        return ISALayer(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


@HEADS.register_module()
class DAFormerHead_Graph(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(DAFormerHead_Graph, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert not self.align_corners
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)
        
        self.cross_domain_graph = MultiHeadAttention_Graph(256, 1, dropout=0.1, version='v2') # Cross Graph Interaction
        self.intra_domain_graph = MultiHeadAttention_Graph(256, 1, dropout=0.1, version='v2') # Intra-domain graph aggregation

        self.register_buffer('sr_seed', torch.zeros(self.num_classes, 256))
        self.register_buffer('tg_seed', torch.zeros(self.num_classes, 256))

        self.InstNorm_layer = nn.InstanceNorm2d(1)
        self.matching_loss = nn.MSELoss(reduction='mean')
        self.node_affinity = Affinity(d=256)


        # euler-based attention block
        self.euler_margin_att = Euler_Attention(
            n_heads=2,
            hidden_size=self.channels,
            hidden_dropout_prob=0.1,
            attn_dropout_prob=0.1,
        )

    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        x_feat=x

        B, C, H, W = x.shape
        downsampled_features = nn.AdaptiveAvgPool2d((H // 2, W // 2))(x)
        B_ds, C_ds, H_ds, W_ds = downsampled_features.shape
        fused_3d = downsampled_features.flatten(2).transpose(1, 2)  # B, L_ds, C
        fused_3d = self.euler_margin_att(fused_3d)
        refined_downsampled = fused_3d.transpose(1, 2).reshape(B_ds, C_ds, H_ds, W_ds) # B, C, H_ds, W_ds
        refined_fused = resize(
            refined_downsampled, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=self.align_corners
        )
        x = x + refined_fused

        x = self.cls_seg(x)
        return x,x_feat

    def eu_dis(self, a,b,p=2):
        return torch.norm(a[:,None]-b,dim=2,p=p)

    def one_hot(self, x):
        device = x.device
        return torch.eye(self.num_classes, device=device)[x.long(), :]

    def sinkhorn_rpm(self, log_alpha, n_iters=5, slack=True, eps=-1):
        ''' Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

        Args:
            log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
            n_iters (int): Number of normalization iterations
            slack (bool): Whether to include slack row and column
            eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K)

        Modified from original source taken from:
            Learning Latent Permutations with Gumbel-Sinkhorn Networks
            https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
        '''
        prev_alpha = None
        if slack:
            zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
            log_alpha_padded = zero_pad(log_alpha[:, None, :, :])
            log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

            for i in range(n_iters):
                # Row normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                    dim=1)
                # Column normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                    dim=2)
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()
            log_alpha = log_alpha_padded[:, :-1, :-1]
        else:
            for i in range(n_iters):
                # Row normalization (i.e. each row sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))
                # Column normalization (i.e. each column sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha).clone()              
        return log_alpha

    def _forward_qu(self, edge_1, edge_2, affinity):
        R =  torch.mm(edge_1, affinity) - torch.mm(affinity, edge_2)
        loss = torch.nn.L1Loss(reduction='mean')(R, R.new_zeros(R.size()))
        return loss
    
    def _forward_aff(self, nodes_1, nodes_2, labels_side1, labels_side2):

        M = self.node_affinity(nodes_1, nodes_2)
        M = self.InstNorm_layer(M[None, None, :, :])
        M = self.sinkhorn_rpm(M[:, 0, :, :]/0.05, n_iters=20, eps=1e-3).squeeze().exp()
        one_hot1 = self.one_hot(labels_side1)
        one_hot2 = self.one_hot(labels_side2)
        matching_target = torch.mm(one_hot1, one_hot2.t())

        unknown_class = self.num_classes - 1
        mask_unknown_1 = (labels_side1 == unknown_class)
        mask_unknown_2 = (labels_side2 == unknown_class)
        matching_target[mask_unknown_1, :] = 0
        matching_target[:, mask_unknown_2] = 0

        # True-positive mask
        TP_mask = (matching_target == 1).float()
        TP_samples = (M * TP_mask)[TP_mask.bool()]
        FP_samples = (M * (1 - TP_mask))[~TP_mask.bool()]

        if TP_samples.numel() == 0 or FP_samples.numel() == 0:
            return torch.tensor(0.0, device=M.device), M

        TP_target = torch.ones_like(TP_samples)
        FP_target = torch.zeros_like(FP_samples)
        loss_TP = nn.MSELoss(reduction='mean')(TP_samples, TP_target.float())
        loss_FP = nn.MSELoss(reduction='mean')(FP_samples, FP_target.float())

        matching_loss = (loss_TP + loss_FP)
        unk_id=self.num_classes - 1
        mask_known_1 = labels_side1 != unk_id
        mask_known_2 = labels_side2 != unk_id
        mask_unk1 = labels_side1 == unk_id
        mask_unk2 = labels_side2 == unk_id

        if mask_known_1.any() and mask_unk1.any():
            nodes_1_known = F.normalize(nodes_1[mask_known_1], dim=-1)
            nodes_1_unk   = F.normalize(nodes_1[mask_unk1],   dim=-1)
            sim_known_unk = torch.matmul(nodes_1_known, nodes_1_unk.t())
            loss_sep_1 = torch.norm(sim_known_unk, p='fro')** 2 / sim_known_unk.numel()
        else:
            loss_sep_1 = torch.tensor(0.0, device=M.device)

        if mask_unk2.any() and mask_known_2.any():
            nodes_2_unk   = F.normalize(nodes_2[mask_unk2],   dim=-1)
            nodes_2_known = F.normalize(nodes_2[mask_known_2], dim=-1)
            sim_unk_known = torch.matmul(nodes_2_unk, nodes_2_known.t())
            loss_sep_2 = torch.norm(sim_unk_known, p='fro')** 2 / sim_unk_known.numel()
        else:
            loss_sep_2 = torch.tensor(0.0, device=M.device)

        loss = matching_loss+(loss_sep_1 + loss_sep_2)*0.5
        return loss, M

   
    def update_seed(self, sr_nodes, sr_labels, tg_nodes=None, tg_labels=None):

        def process_nodes_and_labels(nodes, labels, seed):
            labels = labels.squeeze().reshape(-1).long()
            for cls in labels.unique().long():
                bs = nodes[labels == cls].detach()
                if bs.numel() == 0:
                    continue
                bs_mean = bs.mean(0)
                seed[cls] = 0.9 * seed[cls] + 0.1 * bs_mean
        process_nodes_and_labels(sr_nodes, sr_labels, self.sr_seed)
        process_nodes_and_labels(tg_nodes, tg_labels, self.tg_seed)

    def update_seed_init(self, sr_nodes, sr_labels, tg_nodes=None, tg_labels=None):
        def process_nodes_and_labels(nodes, labels, seed):
            labels = labels.squeeze().reshape(-1).long()
            for cls in labels.unique().long():
                bs = nodes[labels == cls].detach()
                if bs.numel() == 0:
                    continue
                bs_mean = bs.mean(0)
                seed[cls] = 0.5 * seed[cls] + 0.5 * bs_mean
        process_nodes_and_labels(sr_nodes, sr_labels, self.sr_seed)
        process_nodes_and_labels(tg_nodes, tg_labels, self.tg_seed)
   
    def node_sample(self, x_feat, seg_logits, gt_x_feat, max_nodes_per_class=8):
        """
        Node sampling with class balance, positive/negative balance,
        Args:
            x_feat: [B, C, H, W] feature map.
            seg_logits: [B, num_classes, H, W] segmentation logits.
            gt_x_feat: [B, H, W] ground truth labels.
            max_nodes_per_class: maximum number of nodes per class per positive/negative set.
            intra_mix_alpha: tuple (min, max) for linear interpolation in Intra-Mix.

        Returns:
            source_nodes_feat, target_nodes_feat, source_nodes_gt, target_nodes_gt
        """
        B, C, H, W = x_feat.shape
        assert B % 2 == 0, "Batch size B must be even: first half source, second half target."

        source_feat = x_feat[:B // 2]
        target_feat = x_feat[B // 2:]
        source_gt   = gt_x_feat[:B // 2]
        target_gt   = gt_x_feat[B // 2:]

        source_gt = torch.where(
            source_gt == 255,
            torch.tensor(self.num_classes - 1, device=source_gt.device, dtype=source_gt.dtype),
            source_gt
        )

        def flatten_btchw(feat_4d, gt_3d, logits_4d):
            B2, C2, H2, W2 = feat_4d.shape
            feat_flat = feat_4d.view(B2, C2, -1).permute(0, 2, 1).reshape(-1, C2)
            gt_flat   = gt_3d.reshape(-1)
            logits_flat = logits_4d.permute(0, 2, 3, 1).reshape(-1, logits_4d.size(1))
            probs_flat = F.softmax(logits_flat, dim=1)
            max_prob, _ = probs_flat.max(dim=1)
            entropy_flat = -torch.sum(probs_flat * F.log_softmax(logits_flat, dim=1), dim=1)
            return feat_flat, gt_flat, max_prob, entropy_flat

        s_feat_flat, s_gt_flat, s_prob_flat, s_entropy_flat = flatten_btchw(source_feat, source_gt, seg_logits[:B//2])
        t_feat_flat, t_gt_flat, t_prob_flat, t_entropy_flat = flatten_btchw(target_feat, target_gt, seg_logits[B//2:])

        known_classes = list(range(self.num_classes))

        source_feat_list, source_gt_list = [], []
        target_feat_list, target_gt_list = [], []

        def _select_nodes_for_class(feat_flat, gt_flat, prob_flat, entropy_flat, cls_id, is_positive):
            """Node selection with class and positive/negative balance"""
            if cls_id >= self.num_classes - 1:  # unknown
                cls_mask = gt_flat == cls_id
                if not torch.any(cls_mask):
                    return None, None

                cls_feats = feat_flat[cls_mask]
                cls_probs = prob_flat[cls_mask]
                cls_entropy = entropy_flat[cls_mask]
                unk_feats_std = cls_feats.std().item()
                mean_feat = cls_feats.mean(dim=0, keepdim=True)

                prob_thr =  cls_probs.quantile(0.5).item()
                entropy_thr = cls_entropy.quantile(0.5).item()

                if is_positive:
                    unk_mask = (cls_probs>prob_thr)&(cls_entropy < entropy_thr)
                    unk_feats = cls_feats[unk_mask]
                else:
                    unk_mask = (cls_probs<prob_thr)&(cls_entropy > entropy_thr)
                    unk_feats = cls_feats[unk_mask]
                
                if not torch.any(unk_mask):
                    return None, None

                dist = torch.norm(unk_feats - mean_feat, dim=1)
                topk = min(max_nodes_per_class, unk_feats.size(0)) 
                nearest_idx = torch.topk(dist, k=topk, largest=False).indices
                nodes = []
                mean_nodes = mean_feat + torch.normal(
                    mean=0.0,
                    std=unk_feats_std/2,
                    size=(1, cls_feats.size(1)),
                    device=cls_feats.device,
                )
                nodes.append(mean_nodes)

                nodes.append(unk_feats[nearest_idx])

                unk_nodes = torch.cat(nodes, dim=0)
                unk_gt = torch.full(
                    (unk_nodes.size(0),), self.num_classes - 1,
                    device=feat_flat.device, dtype=torch.long
                )

                return unk_nodes, unk_gt

            else:
                cls_mask = gt_flat == cls_id
                if not torch.any(cls_mask):
                    return None, None

                cls_feats = feat_flat[cls_mask]
                cls_probs = prob_flat[cls_mask]
                cls_entropy = entropy_flat[cls_mask]
                mean_feat_std=cls_feats.std().item()
                mean_feat = cls_feats.mean(dim=0, keepdim=True)
                prob_thr = cls_probs.quantile(0.5).item() 
                entropy_thr = cls_entropy.quantile(0.5).item()

                if is_positive:
                    mask = (cls_probs > prob_thr) & (cls_entropy < entropy_thr)
                else:
                    mask = (cls_probs <= prob_thr) & (cls_entropy < entropy_thr)

                if not torch.any(mask):
                    return None, None
                
                if max_nodes_per_class > cls_feats.size(0):
                    mean_num   = max_nodes_per_class-cls_feats.size(0)+max_nodes_per_class
                else:
                    mean_num   = max_nodes_per_class

                cls_feats = cls_feats[mask]
                cls_probs = cls_probs[mask]
                num_select = min(max_nodes_per_class, cls_feats.size(0))

                nodes = [mean_feat+ torch.normal(0, mean_feat_std/2, size=(mean_num, cls_feats.size(1)), device=cls_feats.device)]
                if num_select > 1:
                    dists = torch.cdist(cls_feats, mean_feat, p=2).squeeze(1)
                    sorted_indices = torch.topk(dists, k=num_select, largest=False).indices
                    nodes.append(cls_feats[sorted_indices])

                feat_nodes = torch.cat(nodes, dim=0)
                gt_nodes = torch.full((feat_nodes.size(0),), cls_id,
                                    device=gt_flat.device, dtype=gt_flat.dtype)
                return feat_nodes, gt_nodes

        for c in known_classes:
            for is_positive in [True, False]:
                s_feat, s_gt = _select_nodes_for_class(s_feat_flat, s_gt_flat, s_prob_flat, s_entropy_flat, c, is_positive)
                t_feat, t_gt = _select_nodes_for_class(t_feat_flat, t_gt_flat, t_prob_flat, t_entropy_flat, c, is_positive)

                if s_feat is not None:
                    source_feat_list.append(s_feat)
                    source_gt_list.append(s_gt)
                if t_feat is not None:
                    target_feat_list.append(t_feat)
                    target_gt_list.append(t_gt)

        def _safe_cat(tlist, dim=0, empty_shape=(0, C)):
            return torch.cat(tlist, dim=dim) if tlist else x_feat.new_zeros(empty_shape)

        source_nodes_feat = _safe_cat(source_feat_list)
        source_nodes_gt   = _safe_cat(source_gt_list, empty_shape=(0,))
        target_nodes_feat = _safe_cat(target_feat_list)
        target_nodes_gt   = _safe_cat(target_gt_list, empty_shape=(0,))

        return source_nodes_feat, target_nodes_feat, source_nodes_gt, target_nodes_gt


    def _node_completion(self, nodes, labels):
        sr_nodes, tg_nodes = nodes
        sr_labels, tg_labels = labels

        labels_exist = torch.cat([sr_labels, tg_labels]).unique()

        sr_nodes_all, tg_nodes_all = [sr_nodes], [tg_nodes]
        sr_labels_all, tg_labels_all = [sr_labels], [tg_labels]

        for c in labels_exist:
            sr_mask, tg_mask = sr_labels == c, tg_labels == c
            has_sr, has_tg = sr_mask.any(), tg_mask.any()

            sr_c = sr_nodes[sr_mask]
            tg_c = tg_nodes[tg_mask]

            if has_sr and has_tg:
                continue
            
            elif has_tg and not has_sr:
                num = len(tg_c)//4
                sr_c_fake = (
                    torch.normal(0,tg_c.std().item()/2, size=(num, sr_nodes.size(1)), device=tg_c.device)+ self.sr_seed[c]
                )

                sr_nodes_all.append(sr_c_fake)
                sr_labels_all.append(torch.full((num,), c, dtype=torch.long, device=tg_c.device))

            elif has_sr and not has_tg:
                num = len(sr_c)//4
                tg_c_fake = (
                    torch.normal(0, sr_c.std().item()/2, size=(num, tg_nodes.size(1)), device=sr_c.device)+ self.tg_seed[c]
                )
                tg_nodes_all.append(tg_c_fake)
                tg_labels_all.append(torch.full((num,), c, dtype=torch.long, device=sr_c.device))

        return (
            torch.cat(sr_nodes_all, dim=0),
            torch.cat(tg_nodes_all, dim=0)
        ), (
            torch.cat(sr_labels_all, dim=0),
            torch.cat(tg_labels_all, dim=0)
        )


    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      match=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if match==None or match==False:
            seg_logits,x_feat = self.forward(inputs)
            losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        elif match==True:
            # Forward pass
            seg_logits, x_feat = self.forward(inputs)
            B, C, H, W = x_feat.shape

            # Resize ground truth segmentation
            gt_x_feat = F.interpolate(gt_semantic_seg.float(),
                                    size=(H, W),
                                    mode='nearest').squeeze(1).long()

            # Node sampling
            (
                pos_src_feat, pos_tgt_feat,
                pos_src_gt, pos_tgt_gt,
            ) = self.node_sample(x_feat, seg_logits, gt_x_feat)
            # Initialize losses
            losses = {}
            losses['loss_matchgraph'] = torch.tensor(0.0, requires_grad=True).to(x_feat.device)
            sr_seed_all = torch.all(self.sr_seed.norm(dim=-1) > 0)
            tg_seed_all = torch.all(self.tg_seed.norm(dim=-1) > 0)
            if sr_seed_all and tg_seed_all:
                # Node completion
                (pos_src_feat, pos_tgt_feat), (pos_src_gt, pos_tgt_gt)= \
                    self._node_completion((pos_src_feat, pos_tgt_feat),
                                        (pos_src_gt, pos_tgt_gt))
                # # Intra-domain graph
                pos_src_feat, pos_edges_s = self.intra_domain_graph(pos_src_feat, pos_src_feat, pos_src_feat)
                pos_tgt_feat, pos_edges_t = self.intra_domain_graph(pos_tgt_feat, pos_tgt_feat, pos_tgt_feat)
                self.update_seed(pos_src_feat, pos_src_gt, pos_tgt_feat, pos_tgt_gt)
                pos_tgt_feat = self.cross_domain_graph(pos_src_feat.detach(), pos_src_feat.detach(), pos_tgt_feat)[0]
                pos_src_feat= self.cross_domain_graph(pos_tgt_feat.detach(), pos_tgt_feat.detach(), pos_src_feat)[0]
                match_loss_pos, affinity = self._forward_aff(pos_src_feat, pos_tgt_feat, pos_src_gt, pos_tgt_gt)
                loss_quadratic = self._forward_qu(pos_edges_s.detach(), pos_edges_t.detach(), affinity)
                losses['loss_matchgraph'] = (match_loss_pos+loss_quadratic) * 0.1
            else:
                self.update_seed_init(pos_src_feat, pos_src_gt,pos_src_feat, pos_src_gt)
        
        return losses
    
    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)[0]

    
