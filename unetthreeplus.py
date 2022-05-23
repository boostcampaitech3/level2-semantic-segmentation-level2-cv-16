from typing import Optional, Union, List
from requests import head

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationModel, ClassificationHead
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base.modules import Activation

import torch
import torch.nn as nn
import torch.nn.functional as F




class DownBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size,kernel_size, ceil_mode=True)
        self.conv = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention = md.Attention(attention_type, in_channels=out_channels)
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        x = self.attention(x)
        return x

class UpperBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            scale_factor = 2,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.conv = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention = md.Attention(attention_type, in_channels=out_channels)
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.attention(x)
        return x


class SkipBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class Unet3PlusDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            cat_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()
        if cat_channels*n_blocks != decoder_channels:
            raise ValueError(
                "n_block is {} skip_channels is {}, but you provide decoder_channels for {} blocks.".format(
                    n_blocks, cat_channels, decoder_channels
                )
            )
        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        head_channels = encoder_channels[-1]
        self.n_blocks = n_blocks
        self.cat_channels = cat_channels
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(self.n_blocks - 1):
            for start_idx in range(self.n_blocks):
                if layer_idx >= start_idx:
                    in_ch = encoder_channels[start_idx]
                    out_ch = self.cat_channels
                    if layer_idx == start_idx:
                        blocks[f'x_{layer_idx}_{start_idx}'] = SkipBlock(in_ch, out_ch, **kwargs)
                    else:
                        blocks[f'x_{layer_idx}_{start_idx}'] = DownBlock(in_ch, out_ch, kernel_size = 2**(layer_idx - start_idx), **kwargs)
                elif start_idx == self.n_blocks -1:
                    blocks[f'x_{layer_idx}_{start_idx}'] = UpperBlock(head_channels, out_ch, scale_factor = 2**(start_idx - layer_idx) ,**kwargs)
                else:
                    in_ch = self.out_channels
                    out_ch = self.cat_channels
                    blocks[f'x_{layer_idx}_{start_idx}'] = UpperBlock(in_ch, out_ch, scale_factor = 2**(start_idx - layer_idx), **kwargs)
            blocks[f'x_{layer_idx}'] = SkipBlock(self.out_channels, self.out_channels, **kwargs)
        self.blocks = nn.ModuleDict(blocks)

    def forward(self, *features):

        #features = features[1:]    # remove first skip with same spatial resolution
        # start building dense connections
        features = features[1:]    # remove first skip with same spatial resolution
        dense_x = {}
        dense_x[f'x_{self.n_blocks-1}'] = features[-1]
        for layer_idx in range(self.n_blocks-2,- 1,-1):
            cat_features = []
            for start_idx in range(self.n_blocks-1,-1,- 1):
                
                if layer_idx >= start_idx:
                    cat_features.append(self.blocks[f'x_{layer_idx}_{start_idx}'](features[start_idx]))
                else:
                    cat_features.append(self.blocks[f'x_{layer_idx}_{start_idx}'](dense_x[f'x_{start_idx}']))
            dense_x[f'x_{layer_idx}'] = self.blocks[f'x_{layer_idx}'](torch.cat(cat_features, dim = 1))
        return dense_x['x_0'],dense_x['x_1'],dense_x['x_2'],dense_x['x_3']

class SegmentationHead(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=(4,8,16,32)):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.ups0 = nn.UpsamplingBilinear2d(scale_factor=upsampling[0]) if upsampling[0] > 1 else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.ups1 = nn.UpsamplingBilinear2d(scale_factor=upsampling[1]) if upsampling[1] > 1 else nn.Identity()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.ups2 = nn.UpsamplingBilinear2d(scale_factor=upsampling[2]) if upsampling[2] > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(2048, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.ups3 = nn.UpsamplingBilinear2d(scale_factor=upsampling[3]) if upsampling[3] > 1 else nn.Identity()
        self.activation = Activation(activation)
    def forward(self,x):
        return [self.ups0(self.conv0(x[0])),self.ups1(self.conv1(x[1])),self.ups2(self.conv2(x[2])),self.ups3(self.conv3(x[3]))]

class Unet3Plus(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "tu-beit_base_patch16_384",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: int = 320,
        ds = False,
        cat_channels: int = 64,
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        self.ds= ds
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=None
        )

        self.decoder = Unet3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            cat_channels = cat_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling = (4,8,16,32)
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "unet3plus-{}".format(encoder_name)
        self.initialize()