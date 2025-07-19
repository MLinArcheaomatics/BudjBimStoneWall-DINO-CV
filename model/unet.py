import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        feature_channels: List[int], 
        n_classes: int, 
        concat_mult: int=2, 
        bilinear: bool=False, 
        dropout_rate: float=0.0
    ):
        """
        U-Net-like segmentation network with a customizable encoder.

        Args:
            encoder (torch.nn.Module): Pre-trained encoder network.
            feature_channels (list[int]): Number of channels at each feature level from the encoder.
            n_classes (int): Number of output classes.
            concat_mult (int, optional): Multiplier for concatenated features. Defaults to 2.
            bilinear (bool, optional): Use bilinear interpolation for upsampling. Defaults to False.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.0.
        """
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.interpolate_mode = 'bilinear' if bilinear else 'nearest'
        self.feature_channels = feature_channels
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Create layers dynamically based on feature hierarchy
        for i in range(0, len(feature_channels) - 1):
            in_channel = feature_channels[i + 1] * concat_mult
            setattr(self, 'shrink%d' % i,
                    nn.Conv2d(in_channel, feature_channels[i] * concat_mult, kernel_size=3, stride=1, padding=1))
            setattr(self, 'shrink2%d' % i,
                    nn.Conv2d(feature_channels[i] * concat_mult * 2, feature_channels[i] * concat_mult, kernel_size=3, stride=1, padding=1, bias=False))
            setattr(self, 'batchnorm%d' % i,
                    nn.BatchNorm2d(feature_channels[i] * concat_mult))
            
        self.outc = OutConv(feature_channels[0] * concat_mult, n_classes)
        self.encoder = encoder

    def forward(self, *in_x):
        features = self.encoder(*in_x)
        features = features[1:]
        x = features[-1]
        
        for i in range(len(features) - 2, -1, -1):
            conv = getattr(self, 'shrink%d' % i)
            x = F.interpolate(x, scale_factor=2, mode=self.interpolate_mode)
            x = conv(x)
            
            if features[i].shape[-1] != x.shape[-1]:
                x2 = F.interpolate(features[i], scale_factor=4, mode=self.interpolate_mode)
            else:
                x2 = features[i]
                
            x = torch.cat([x, x2], 1)
            conv2 = getattr(self, 'shrink2%d' % i)
            x = conv2(x)
            bn = getattr(self, 'batchnorm%d' % i)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final upsampling and output
        x = F.interpolate(x, scale_factor=2, mode=self.interpolate_mode)
        logits = self.outc(x)
        return logits
    

class SiameseUNet(nn.Module):
    def __init__(
        self, 
        encoders: List[nn.Module], 
        feature_channels: List[int], 
        n_classes: int, 
        concat_mult: int=2, 
        bilinear: bool=False, 
        dropout_rate: float=0.0
    ):
        """
        Siamese U-Net-like segmentation network with a customizable encoders.

        Args:
            encoders (list[torch.nn.Module]): list of Pre-trained encoder networks.
            feature_channels (list[int]): Number of channels at each feature level from the encoder.
            n_classes (int): Number of output classes.
            concat_mult (int, optional): Multiplier for concatenated features. Defaults to 2.
            bilinear (bool, optional): Use bilinear interpolation for upsampling. Defaults to False.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.0.
        """
        super(SiameseUNet, self).__init__()
        self.n_classes = n_classes
        self.interpolate_mode = 'bilinear' if bilinear else 'nearest'
        self.feature_channels = feature_channels
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Create layers dynamically based on feature hierarchy
        for i in range(0, len(feature_channels) - 1):
            in_channel = feature_channels[i + 1] * concat_mult
            setattr(self, 'shrink%d' % i,
                    nn.Conv2d(in_channel, feature_channels[i] * concat_mult, kernel_size=3, stride=1, padding=1))
            setattr(self, 'shrink2%d' % i,
                    nn.Conv2d(feature_channels[i] * concat_mult * 2, feature_channels[i] * concat_mult, kernel_size=3, stride=1, padding=1, bias=False))
            setattr(self, 'batchnorm%d' % i,
                    nn.BatchNorm2d(feature_channels[i] * concat_mult))
            
        self.outc = OutConv(feature_channels[0] * concat_mult, n_classes)
        self.encoders = nn.ModuleList(encoders)

    def forward(self, inputs):
        encoder_features = []
        for input, encoder in zip (inputs, self.encoders):
            encoder_features.append(encoder(input)[1:])
            
        xs = []
        for feature in list(zip(*encoder_features)):
            xs.append(torch.stack(feature).sum(dim=0))

        x = xs[-1]
        
        for i in range(len(xs) - 2, -1, -1):
            conv = getattr(self, 'shrink%d' % i)
            x = F.interpolate(x, scale_factor=2, mode=self.interpolate_mode)
            x = conv(x)
            
            if xs[i].shape[-1] != x.shape[-1]:
                x2 = F.interpolate(xs[i], scale_factor=4, mode=self.interpolate_mode)
            else:
                x2 = xs[i]
                
            x = torch.cat([x, x2], 1)
            conv2 = getattr(self, 'shrink2%d' % i)
            x = conv2(x)
            bn = getattr(self, 'batchnorm%d' % i)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final upsampling and output
        x = F.interpolate(x, scale_factor=2, mode=self.interpolate_mode)
        logits = self.outc(x)
        return logits