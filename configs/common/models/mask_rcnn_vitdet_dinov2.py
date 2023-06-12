from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.modeling import VisionTransformer_ViTDet, SimpleFeaturePyramid_ViTDet, SimpleFeaturePyramid, ViT
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .mask_rcnn_fpn import model
from ..data.constants import constants

model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"

patch_size=14
embed_dim=384
depth=12
num_heads=6
init_values=1e-5
img_size=896
# Base
# embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
# Creates Simple Feature Pyramid from ViT backbone

# model.backbone = L(SimpleFeaturePyramid)(
#     net=L(ViT)(  # Single-scale ViT backbone
model.backbone = L(SimpleFeaturePyramid_ViTDet)(
    net=L(VisionTransformer_ViTDet)(  # Single-scale ViT backbone
        init_values=init_values,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        flatten_patch_embed=True,
        output_fmt_patch_embed = None,
        class_token=False,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # window_block_indexes=[
        #     # 2, 5, 8 11 for global attention
        #     0,
        #     1,
        #     3,
        #     4,
        #     6,
        #     7,
        #     9,
        #     10,
        # ],
        # residual_block_indexes=[],
        # use_rel_pos=True,
        out_feature="last_feat",
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=img_size,
)

model.roi_heads.box_head.conv_norm = model.roi_heads.mask_head.conv_norm = "LN"
model.roi_heads.num_classes = 1

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]

model.roi_heads.mask_in_features = None
