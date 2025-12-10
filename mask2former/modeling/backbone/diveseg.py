# Copyright (c) Shanghai AI Lab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn
from functools import partial

import torch.utils.checkpoint as cp
from .backbones import get_models
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

class StyleInjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_a = nn.Linear(dim, dim)
        self.atten = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)

    def forward(self, x, a):
        cls = x[:, :1, :]  # extract CLS token
        x = x[:, 1:, :]
        a_proj = self.proj_a(a)  # (bs, 1, dim)
        x, _ = self.atten(x, a_proj, a_proj)
        return torch.cat([cls, x], dim=1), a_proj  # restore CLS; return projected style token

class Adapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Linear(dim, dim)
        self.active = nn.GELU()
        self.proj_2 = nn.Linear(dim, dim)

    def forward(self, x):
        cls = x[:, :1, :]  # extract CLS token
        x = x[:, 1:, :]
        x = self.proj_1(x)  # (bs, n, dim)
        x = self.active(x)
        x = self.proj_2(x)
        return torch.cat([cls, x], dim=1)  # restore CLS to the front

class StyleExtractor(nn.Module):
    def __init__(self, em_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(3, em_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(em_dim, em_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(em_dim, em_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images_a):
        x = self.conv1(images_a)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.avgpool(x)
        bs, dim, _, _ = x.shape
        x = x.view(bs, dim, -1).transpose(1, 2)
        return x

def get_reference_points(spatial_shapes, device):
    # build normalized reference points for multi-scale deformable attention
    reference_points_list = []
    for H_, W_ in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points

def deform_inputs(x):
    # prepare inputs for MSDeformAttn: reference points, spatial shapes, level start index
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
        dtype=torch.long, device=x.device
    )
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1

class MultiScaleEncoder(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        # self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)

            # c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
            bs, _, _, _ = c1.shape
            dim = 1024
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

class ObjectPriorPrompter(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlockWithCls(nn.Module):
    def __init__(self, dim, block_num, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  init_values=0., deform_ratio=1.0, with_cp=False, with_adapter=False):
        super().__init__()

        self.ObjectPriorPrompter = ObjectPriorPrompter(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.with_adapter = with_adapter
        if with_adapter:
            self.StyleInjection =  StyleInjection(dim)
            self.Adapter = Adapter(dim)

        self.op_heads = nn.ModuleList([nn.Conv2d(dim, 1, 1, 1, 0) for _ in range(3)])
        self.dim = dim
        self.conv2 = nn.Conv2d(dim, dim, 1, 1)
        self.norm2 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, dim, 1, 1)
        self.norm3 = nn.BatchNorm2d(dim)
        self.conv4 = nn.Conv2d(dim, dim, 1, 1)
        self.norm4 = nn.BatchNorm2d(dim)

    def forward(self, x, c, cls, a, blocks, deform_inputs1, H, W):
        B, N, C = c.shape
        n = N // 21
        c2 = c[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        c3 = c[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        c4 = c[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()


        x = torch.cat((cls, x), dim=1)
        if self.with_adapter:
            for idx, blk in enumerate(blocks):
                if idx == 0:
                    # MHA block
                    residual = x
                    x = blk.norm1(x)
                    x = blk.attn(x)
                    x = blk.ls1(x)
                    x = residual + blk.drop_path1(x)
                    x_StyleInjection, a = self.StyleInjection(residual, a)
                    x = x + x_StyleInjection
                    # FFN block
                    residual = x
                    x = blk.norm2(x)
                    x = blk.mlp(x)
                    x = blk.ls2(x)
                    x = residual + blk.drop_path1(x)
                    x = x + self.Adapter(residual)
                else:
                    x = blk(x)
        else:
            for idx, blk in enumerate(blocks):
                x = blk(x)

        cls, x = x[:, :1, ], x[:, 1:, ]

        op2, op3, op4 = [torch.sigmoid(head(feat)) for head, feat in zip(self.op_heads, [c2, c3, c4])]

        c2_withop = self.norm2(self.conv2(c2 * op2) + c2)
        c3_withop = self.norm3(self.conv3(c3 * op3) + c3)
        c4_withop = self.norm4(self.conv4(c4 * op4) + c4)

        c2_withop = c2_withop.flatten(2).transpose(1, 2)
        c3_withop = c3_withop.flatten(2).transpose(1, 2)
        c4_withop = c4_withop.flatten(2).transpose(1, 2)

        # upsample object prior maps to the same spatial size and concatenate
        op_out = torch.cat([
            F.interpolate(op2, (H, W), mode='bilinear', align_corners=False),
            F.interpolate(op3, (H, W), mode='bilinear', align_corners=False),
            F.interpolate(op4, (H, W), mode='bilinear', align_corners=False)
        ], dim=1)

        c = torch.cat([c2_withop, c3_withop, c4_withop], dim=1)

        x = self.ObjectPriorPrompter(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )

        return x, c, cls, op_out



def get_diveseg_args(
    name='vitl', backbone_weight=None, freeze_backbone=False, finetune=False, finetune_indexes=[0, ], with_cp=False
):
    if freeze_backbone:
        assert backbone_weight is not None
    vit_backbone = get_models(name, backbone_weight)
    bg_configs = vit_backbone.configs_dict
    if name == 'vitl':
        diveseg_args = {
            'vit_module': vit_backbone,
            'pretrain_size': bg_configs['img_size'],
            'init_values': 1e-6,
            'conv_inplane': 64,
            'n_points': 4,
            'deform_num_heads': 16,
            'with_cffn': True,
            'cffn_ratio': 0.25,
            'add_vit_feature': True,
            'use_extra_extractor': False,
            'with_cp': with_cp,
            'interaction_indexes': [[0, 5], [6, 11], [12, 17], [18, 23]],
        }
    elif name == 'vitb':
        diveseg_args = {
            'vit_module': vit_backbone,
            'pretrain_size': bg_configs['img_size'],
            'init_values': 1e-6,
            'conv_inplane': 64,
            'n_points': 4,
            'deform_num_heads': 12,
            'with_cffn': True,
            'cffn_ratio': 0.25,
            'add_vit_feature': True,
            'use_extra_extractor': False,
            'with_cp': with_cp,
            'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
        }
    else:
        raise NotImplementedError
    diveseg_args.update({
        'freeze_backbone': freeze_backbone,
        'finetune': finetune,
        'finetune_indexes': finetune_indexes,
    })
    return diveseg_args


class DiveSeg(nn.Module):
    def __init__(
        self,
        vit_module=None,
        pretrain_size=224,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=False,
        freeze_backbone=True,
        finetune=False,
        finetune_indexes=[0, ],
    ):
        super().__init__()

        self.num_block = len(vit_module.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = vit_module.embed_dim

        self.drop_path_rate = vit_module.configs_dict['drop_path_rate']
        self.norm_layer = vit_module.norm_layer

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.mse = MultiScaleEncoder(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlockWithCls(
                dim=embed_dim,
                block_num=interaction_indexes[i][-1] - interaction_indexes[i][0] + 1,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                norm_layer=self.norm_layer,
                deform_ratio=deform_ratio,
                with_cp=with_cp,
                with_adapter=True,
            )
            for i in range(len(interaction_indexes))
        ])

        self.mse.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)

        self.vit_module = vit_module
        if freeze_backbone:
            for p in self.vit_module.parameters():
                p.requires_grad_(False)
            if finetune:
                finetuned_interaction_indexes = finetune_indexes
                self.finetuned_interaction_indexes = finetuned_interaction_indexes
                for idx in finetuned_interaction_indexes:
                    indexes = self.interaction_indexes[idx]
                    for p in self.vit_module.blocks[indexes[0]:indexes[-1] + 1].parameters():
                        p.requires_grad_(True)

        self.freeze_backbone = freeze_backbone
        self.finetune = finetune

        self.style_extractor = StyleExtractor(embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x, a):
        a = self.style_extractor(a)
        if self.freeze_backbone:
            self.vit_module.eval()
            if self.finetune and self.training:
                # self.vit_module.patch_embed.train()
                finetuned_interaction_indexes = self.finetuned_interaction_indexes
                for idx in finetuned_interaction_indexes:
                    indexes = self.interaction_indexes[idx]
                    self.vit_module.blocks[indexes[0]:indexes[-1] + 1].train()
        deform_inputs1 = deform_inputs(x)

        # mse forward
        c1, c2, c3, c4 = self.mse(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        if self.finetune:
            x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        else:
            with torch.no_grad():
                x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        bs, n, dim = x.shape
        cls = x[:, :1, :]
        x = x[:, 1:, :]

        # Interaction
        outs = list()
        ops = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c, cls, op = layer(
                x, c, cls, a, self.vit_module.blocks[indexes[0]:indexes[-1] + 1], deform_inputs1, H, W
            )
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())
            ops.append(op)

        x1, x2, x3, x4 = outs
        x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)

        op_out = torch.cat(ops, dim=1)

        return [x1, x2, x3, x4], op_out

@BACKBONE_REGISTRY.register()
class DiveSeg_Backbone(DiveSeg, Backbone):
    def __init__(self, cfg, input_shape):
        name = cfg.MODEL.DIVESEG.NAME
        backbone_weight = cfg.MODEL.DIVESEG.VIT_WEIGHT
        freeze_backbone = cfg.MODEL.DIVESEG.FREEZE_VIT
        finetune = cfg.MODEL.DIVESEG.FINETUNE
        finetune_indexes = cfg.MODEL.DIVESEG.FINETUNE_INDEXES
        with_cp = cfg.MODEL.DIVESEG.WITH_CP
        diveseg_args = get_diveseg_args(
            name=name,
            backbone_weight=backbone_weight,
            freeze_backbone=freeze_backbone,
            finetune=finetune,
            finetune_indexes=finetune_indexes,
            with_cp=with_cp,
        )
        super().__init__(**diveseg_args)
        self._out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {
            "res2": self.vit_module.embed_dim,
            "res3": self.vit_module.embed_dim,
            "res4": self.vit_module.embed_dim,
            "res5": self.vit_module.embed_dim,
        }

    def forward(self, x, a):
        assert x.dim() == 4, f"Input must be of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y, op_out = super().forward(x, a)
        for i, k in enumerate(self._out_feature_strides.keys()):
            outputs[k] = y[i]
        return outputs, op_out

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self.vit_module.embed_dim, stride=self._out_feature_strides[name])
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32