"""
Code adapted from Vision Transformers for Dense Prediction

@inproceedings{ranftl2021vision,
  title={Vision transformers for dense prediction},
  author={Ranftl, Ren{\'e} and Bochkovskiy, Alexey and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={12179--12188},
  year={2021}
}

https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/dpt_head.py

and modified from

https://github.com/LiheYoung/UniMatch-V2/blob/main/model/semseg/dpt.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List    


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        
        output = self.out_conv(output)

        return output


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )
    
    
class DPTHead(nn.Module):
    def __init__(
        self, 
        num_classes: int, 
        patch_size: int ,
        embed_dims: int, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],
        bilinear: bool=True
    ):
        super(DPTHead, self).__init__()
        
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.bilinear = bilinear
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=embed_dims,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, num_classes, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, inputs, im_size):
        h, w = im_size[0] // self.patch_size, im_size[1] // self.patch_size
        
        out = []
        for i, x in enumerate(inputs):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], h, w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv(path_1)
        out = F.interpolate(out, size=im_size, mode='bilinear' if self.bilinear else 'nearest', align_corners=True)
        return out
    
    
class DPTSeg(nn.Module):
    """
    ViT for dense prediction.
    """
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        patch_size: int,
        bilinear: bool=True
    ):
        super().__init__()
        embed_dim = encoder.embed_dim
        self.encoder = encoder
        self.decoder = DPTHead(n_classes, patch_size, embed_dim, use_bn=True, bilinear=bilinear)
        
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.bilinear = bilinear

    def forward(self, x):
        """
        Forward pass through the DPT.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Segmentation mask of shape (B, n_classes, H, W).
        """
        H, W = x.size(2), x.size(3)
        x = self.encoder(x)
        masks = self.decoder(x, (H, W))
        return masks
    
    
class SiameseDPTSeg(nn.Module):
    """
    Siamese ViT for dense prediction.
    """
    def __init__(
        self,
        encoders: List[nn.Module], 
        n_classes: int,
        patch_size: int,
        bilinear: bool=True
    ):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.decoder = DPTHead(n_classes, patch_size, self.encoders[0].embed_dim, use_bn=True, bilinear=bilinear)
        
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.bilinear = bilinear

    def forward(self, inputs):
        """
        Forward pass through the DPT.

        Args:
            inputs (List): Input list of image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Segmentation mask of shape (B, n_classes, H, W).
        """
        encoder_features = []
        for input, encoder in zip (inputs, self.encoders):
            encoder_features.append(encoder(input))
            
        xs = []
        for feature in list(zip(*encoder_features)):
            xs.append(torch.stack(feature).sum(dim=0))
        
        H, W = inputs[0].size(2), inputs[0].size(3)
        masks = self.decoder(xs, (H, W))
        return masks