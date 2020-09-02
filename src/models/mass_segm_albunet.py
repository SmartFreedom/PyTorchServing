from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

from ..models import se_models as se
from ..models import deformable_v2 as df

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels, out_channels, 
                kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Interpolate(nn.Module):
    def __init__(self, size=None, 
                 scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        
    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor, 
                        mode=self.mode, align_corners=self.align_corners)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels, out_channels, 
                    kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, head_num_filters=512, 
                 num_head_classes=1, num_filters=32, encoder=None,
                 pretrained=False, is_deconv=False, dropout=0,
                 base_num_filters=512
                ):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        

        self.encoder = encoder
        if encoder is None:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        try:
            self.initial = self.encoder.layer0
        except:
            self.initial = self.encoder
        try: self.pool = self.initial.pool
        except: self.pool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv1 = nn.Sequential(self.initial.conv1,
                                   self.initial.bn1,
                                   nn.ReLU(inplace=True),
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(
            base_num_filters, num_filters * 8 * 2, 
            num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(
            base_num_filters + num_filters * 8, 
            num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(
            base_num_filters // 2 + num_filters * 8, 
            num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(
            base_num_filters // 4 + num_filters * 8,
            num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(
            base_num_filters // 8 + num_filters * 2, 
            num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = nn.Sequential(
            ConvRelu(num_filters * 2 * 2, num_filters * 2 * 2),
            ConvRelu(num_filters * 2 * 2, num_filters),
        )
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.dropout2d = nn.Dropout2d(p=dropout) if dropout else None
        self.head_final = nn.Conv2d(head_num_filters, num_head_classes, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.dropout2d is not None:
            conv2 = self.dropout2d(conv2)
        conv3 = self.conv3(conv2)
        if self.dropout2d is not None:
            conv3 = self.dropout2d(conv3)
        conv4 = self.conv4(conv3)
        if self.dropout2d is not None:
            conv4 = self.dropout2d(conv4)
        conv5 = self.conv5(conv4)
        if self.dropout2d is not None:
            conv5 = self.dropout2d(conv5)
        
        head = self.head_final(conv5)
        center = self.center(self.pool(conv5))
        if self.dropout2d is not None:
            center = self.dropout2d(center)

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        x_out = self.final(dec0)

        return x_out, head

    
class AlbuNetSE(AlbuNet):

    def forward(self, x):
        conv1 = self.conv1(x)

        conv2_, conv2 = self.conv2(conv1)
        conv3_, conv3 = self.conv3(conv2_)
        conv4_, conv4 = self.conv4(conv3_)
        if self.dropout2d is not None:
            conv4 = self.dropout2d(conv4)
        conv5_, conv5 = self.conv5(conv4_)
        if self.dropout2d is not None:
            conv5 = self.dropout2d(conv5)

        center = self.center(self.pool(conv5))
        if self.dropout2d is not None:
            center = self.dropout2d(center)

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        x_out = self.final(dec0)

        return x_out


class DecoderBlockV3(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV3, self).__init__()

        self.block = nn.Sequential(
            Interpolate(scale_factor=2, mode='nearest'),
            # add residual?
            ConvRelu(in_channels, middle_channels),
            nn.Conv2d(middle_channels, out_channels, 1),
#             nn.BatchNorm2d(out_channels)
        )
        self.se_module = se.SEModule(out_channels, 4)

    def forward(self, x):
        x = self.block(x)
        x = self.se_module(x)
        # add residual?
        return x


class FPNDecoderBlock(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, is_deconv=True):
        super(FPNDecoderBlock, self).__init__()

        self.interpolate = Interpolate(
            scale_factor=scale_factor, 
            mode='nearest') if scale_factor != 1 else None
        self.block = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        if self.interpolate is not None:
            x = self.interpolate(x)
        x = self.block(x)
        return x


class FPNAlbuNetSE(AlbuNet):
    def __init__(self, num_classes=1, num_head_classes=1, 
                 head_num_filters=2048, num_filters=32, encoder=None,
                 pretrained=False, is_deconv=False, dropout=0,
                 base_num_filters=512, fpn_channels=256, **kwargs):
        super().__init__(num_classes, num_filters, encoder,
                 pretrained, is_deconv, dropout,
                 base_num_filters, **kwargs)

        # no pooling is applied
        # self.conv1 = nn.Sequential(self.initial.conv1,
        #                            self.initial.bn1,
        #                            nn.ReLU(inplace=True))

        self.center = DecoderBlockV3(
            base_num_filters, num_filters * 8 * 2, 
            fpn_channels, is_deconv)
        self.dec5 = DecoderBlockV3(
            base_num_filters + fpn_channels, 
            num_filters * 8 * 2, fpn_channels, is_deconv)
        self.dec4 = DecoderBlockV3(
            base_num_filters // 2 + fpn_channels, 
            fpn_channels, fpn_channels, is_deconv)
        self.dec3 = DecoderBlockV3(
            base_num_filters // 4 + fpn_channels,
            fpn_channels, fpn_channels, is_deconv)
        self.dec2 = DecoderBlockV3(
            base_num_filters // 8 + fpn_channels, 
            fpn_channels, fpn_channels, is_deconv)

        self.fpn_block5 = FPNDecoderBlock(2**3, fpn_channels, fpn_channels // 2)
        self.fpn_block4 = FPNDecoderBlock(2**2, fpn_channels, fpn_channels // 2)
        self.fpn_block3 = FPNDecoderBlock(2**1, fpn_channels, fpn_channels // 2)
        self.fpn_block2 = FPNDecoderBlock(2**0, fpn_channels, fpn_channels // 2)

        self.dec1 = ConvRelu(fpn_channels * 2, fpn_channels)
        self.final = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)
        
        self.head_final = nn.Conv2d(head_num_filters, num_head_classes, kernel_size=1)

        self.dropout2d = nn.Dropout2d(p=dropout) if dropout else None
    
    def forward(self, x):
        conv1 = self.conv1(x)

        conv2_, conv2 = self.conv2(conv1)
        if self.dropout2d is not None:
            conv2_ = self.dropout2d(conv2_)
        conv3_, conv3 = self.conv3(conv2_)
        if self.dropout2d is not None:
            conv3_ = self.dropout2d(conv3_)
        conv4_, conv4 = self.conv4(conv3_)
        if self.dropout2d is not None:
            conv4_ = self.dropout2d(conv4_)
        conv5_, conv5 = self.conv5(conv4_)
        if self.dropout2d is not None:
            conv5 = self.dropout2d(conv5)

        head = self.head_final(conv5_)

        center = self.center(self.pool(conv5))
        if self.dropout2d is not None:
            center = self.dropout2d(center)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))

        dec2 = self.fpn_block2(dec2)
        dec3 = self.fpn_block3(dec3)
        dec4 = self.fpn_block4(dec4)
        dec5 = self.fpn_block5(dec5)

        dec1 = self.dec1(torch.cat([dec2, dec3, dec4, dec5], 1))

        x_out = self.final(dec1)

        return x_out, head


class DecoderBlockV4(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV4, self).__init__()

        self.block = nn.Sequential(
            df.DeformConv2d(in_channels, middle_channels, 3),
            nn.ReLU(inplace=True),
            Interpolate(scale_factor=2, mode='nearest'),
            nn.Conv2d(middle_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
            # add residual?
        )
        self.se_module = se.SEModule(out_channels, 4)

    def forward(self, x):
        x = self.block(x)
        x = self.se_module(x)
        # add residual?
        return x


class DeformableFPNAlbuNetSE(AlbuNet):
    def __init__(self, num_classes=1, num_filters=32, encoder=None,
                 pretrained=False, is_deconv=False, dropout=0,
                 base_num_filters=512, fpn_channels=256, **kwargs):
        super().__init__(num_classes, num_filters, encoder,
                 pretrained, is_deconv, dropout,
                 base_num_filters, **kwargs)

        # no pooling is applied
        # self.conv1 = nn.Sequential(self.initial.conv1,
        #                            self.initial.bn1,
        #                            nn.ReLU(inplace=True))

        self.center = DecoderBlockV4(
            base_num_filters, num_filters * 8 * 2, 
            fpn_channels, is_deconv)
        self.dec5 = DecoderBlockV4(
            base_num_filters + fpn_channels, 
            num_filters * 8 * 2, fpn_channels, is_deconv)
        self.dec4 = DecoderBlockV4(
            base_num_filters // 2 + fpn_channels, 
            fpn_channels, fpn_channels, is_deconv)
        self.dec3 = DecoderBlockV4(
            base_num_filters // 4 + fpn_channels,
            fpn_channels, fpn_channels, is_deconv)
        self.dec2 = DecoderBlockV3(
            base_num_filters // 8 + fpn_channels, 
            fpn_channels, fpn_channels, is_deconv)

        self.fpn_block5 = FPNDecoderBlock(2**3, fpn_channels, fpn_channels // 2)
        self.fpn_block4 = FPNDecoderBlock(2**2, fpn_channels, fpn_channels // 2)
        self.fpn_block3 = FPNDecoderBlock(2**1, fpn_channels, fpn_channels // 2)
        self.fpn_block2 = FPNDecoderBlock(2**0, fpn_channels, fpn_channels // 2)

        self.dec1 = ConvRelu(fpn_channels * 2, fpn_channels)
        self.final = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)
        self.dropout2d = nn.Dropout2d(p=dropout) if dropout else None
    
    def forward(self, x):
        conv1 = self.conv1(x)

        conv2_, conv2 = self.conv2(conv1)
        if self.dropout2d is not None:
            conv2_ = self.dropout2d(conv2_)
        conv3_, conv3 = self.conv3(conv2_)
        if self.dropout2d is not None:
            conv3_ = self.dropout2d(conv3_)
        conv4_, conv4 = self.conv4(conv3_)
        if self.dropout2d is not None:
            conv4_ = self.dropout2d(conv4_)
        conv5_, conv5 = self.conv5(conv4_)
        if self.dropout2d is not None:
            conv5 = self.dropout2d(conv5)

        center = self.center(self.pool(conv5))
        if self.dropout2d is not None:
            center = self.dropout2d(center)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))

        dec2 = self.fpn_block2(dec2)
        dec3 = self.fpn_block3(dec3)
        dec4 = self.fpn_block4(dec4)
        dec5 = self.fpn_block5(dec5)

        dec1 = self.dec1(torch.cat([dec2, dec3, dec4, dec5], 1))

        x_out = self.final(dec1)

        return x_out
