import torch
from model.unet import UNet, SiameseUNet
from model.dpt import DPTSeg, SiameseDPTSeg


def siamese_segmentation_model(backbones, feature_indices, feature_channels, patch_size, arch, n_classes=1):
    """
    Creates a segmentation model using a pre-trained backbone.

    Args:
        backbones (list[torch.nn.Module]): List of pre-trained backbone models.
        feature_indices (list[int]): Indices of layers in the backbone to extract features.
        feature_channels (list[int]): Number of output channels for each extracted feature.
        patch_size (int): Patch size for transformer-based models (e.g., ViT).
        arch (str): Backbone architecture type, e.g., 'resnet' or 'vit'.
        n_classes (int, optional): Number of output classes for the segmentation task. Defaults to 1.

    Returns:
        torch.nn.Module: Configured segmentation model.
    """
    if 'resnet' in arch:
        backbones = [ResNetEncoder(backbone, feature_indices, feature_channels) for backbone in backbones]
        model = SiameseUNet(
            encoders=backbones,
            feature_channels=feature_channels,
            n_classes=n_classes,
            bilinear=True,
            concat_mult=1,
            dropout_rate=0.5
        )
    elif 'vit' in arch:
        backbones = [VitEncoder(backbone, feature_indices) for backbone in backbones]
        model = SiameseDPTSeg(
            encoders=backbones,
            n_classes=n_classes,
            patch_size=patch_size,
            bilinear=True
        )
    else:
        raise ValueError(f"Unsupported architecture type: {arch}")
    
    return model


def segmentation_model(backbone, feature_indices, feature_channels, patch_size, arch, n_classes=1):
    """
    Creates a segmentation model using a pre-trained backbone.

    Args:
        backbone (torch.nn.Module): Pre-trained backbone model.
        feature_indices (list[int]): Indices of layers in the backbone to extract features.
        feature_channels (list[int]): Number of output channels for each extracted feature.
        patch_size (int): Patch size for transformer-based models (e.g., ViT).
        arch (str): Backbone architecture type, e.g., 'resnet' or 'vit'.
        n_classes (int, optional): Number of output classes for the segmentation task. Defaults to 1.

    Returns:
        torch.nn.Module: Configured segmentation model.
    """
    if 'resnet' in arch:
        model = UNet(
            encoder=ResNetEncoder(backbone, feature_indices, feature_channels),
            feature_channels=feature_channels,
            n_classes=n_classes,
            bilinear=True,
            concat_mult=1,
            dropout_rate=0.5
        )
    elif 'vit' in arch:
        model = DPTSeg(
            encoder=VitEncoder(backbone, feature_indices),
            n_classes=n_classes,
            patch_size=patch_size,
            bilinear=True
        )
    else:
        raise ValueError(f"Unsupported architecture type: {arch}")
    
    return model


class VitEncoder(torch.nn.Module):
    """
    Encoder module for Vision Transformer (ViT) backbones.
    """
    def __init__(self, backbone, feature_indices):
        super().__init__()
        assert hasattr(backbone, 'embed_dim'), 'Backbone must have an "embed_dim" attribute'
        self.embed_dim = backbone.embed_dim
        self.encoder = backbone
        # feature indices of the nth last ViT blocks
        self.feature_indices = sorted(feature_indices)

    def forward(self, x):
        """
        Forward pass for the ViT encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Feature tensor extracted from the encoder of shape (B, L, D),
                          where L is the sequence length and D is the embedding dimension.
        """
        feats = []
        if hasattr(self.encoder, 'forward_intermediates') and callable(self.encoder.forward_intermediates):
            for idx in self.feature_indices:
                intermediate_layers = self.encoder.forward_intermediates(
                    x, 
                    indices=idx, 
                    return_prefix_tokens=False, 
                    output_fmt = 'NLC',
                    intermediates_only=True
                )[0]
                feats.append(intermediate_layers)
            return feats
        elif hasattr(self.encoder, 'get_intermediate_layers') and callable(self.encoder.get_intermediate_layers):
            for idx in self.feature_indices:
                # Exclude the [CLS] token.
                intermediate_layers = self.encoder.get_intermediate_layers(x, n=idx)[0][:, 1:]
                feats.append(intermediate_layers)
            return feats


class ResNetEncoder(torch.nn.Module):
    """
    Encoder module for ResNet-based backbones.
    """
    def __init__(self, backbone, feature_indices, feature_channels):
        """
        Initializes the ResNet encoder.

        Args:
            backbone (torch.nn.Module): ResNet backbone model.
            feature_indices (list[int]): Indices of layers to extract features.
            feature_channels (list[int]): Number of output channels for each extracted feature.
        """
        super().__init__()
        self.feature_indices = sorted(feature_indices)
        self._in_channels = 3
        self._out_channels = feature_channels
        self.encoder = backbone

    def forward(self, x):
        """
        Forward pass for the ResNet encoder.

        Extracts a list of feature maps with descending spatial resolutions.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            list[torch.Tensor]: List of feature maps extracted at specified indices.
        """
        feats = [x]
        for i, module in enumerate(self.encoder.children()):
            x = module(x)
            
            if i in self.feature_indices:
                feats.append(x)
            if i == self.feature_indices[-1]:
                break
        return feats