#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model.stnet import STNet


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.class_num = class_num
        self.dropout_rate = dropout_rate
        self.stn = STNet()
        self.stage1 = nn.Sequential(  # 3x48x96
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # 64x48x96
            nn.MaxPool2d((2, 2), stride=(2, 2), ceil_mode=True),  # 64x24x48
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, groups=64),  # 64x24x48
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=(2, 0), groups=64), # 64x24x44  S
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            small_basic_block(ch_in=64, ch_out=128),  # 128x24x44
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.outconv1 = nn.Sequential(  # 128x24x44 k(5,5) s(5,2)   Stage1
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=(5, 2), groups=128), # 128x4x20  k(3,3) s(1,1)
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.down1 = nn.Sequential(
            small_basic_block(ch_in=128, ch_out=128),  # 128x24x44
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)  # 64x24x44
        )
        self.stage2 = nn.Sequential(  # 64x24x44  k(5,5) s(2,2) p(2,2)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, groups=64),  # 64x12x22  S
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            small_basic_block(ch_in=64, ch_out=128),  # 128x12x22 S
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.outconv2 = nn.Sequential(  # 128x12x22 k(5,3) s(2,1)   Stage2
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 3), stride=(2, 1), groups=128), # 128x4x20  k(3,3) s(1,1)
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            small_basic_block(ch_in=128, ch_out=128),  # 128x12x22
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)  # 128x12x22
        )
        self.stage3 = nn.Sequential(  # 128x12x22
            small_basic_block(ch_in=128, ch_out=128),  # 128x12x22 k(1,1) s(1,1)
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1),  # 256x12x22
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 13), stride=(2, 1), padding=(1, 6), groups=256),  # 256x6x22  k(3,3) s(1,1)
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, groups=256), # 256x4x20  k(3,3) s(1,1)
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, groups=512),  # 512x2x18
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.container2 = nn.Conv2d(in_channels=512, out_channels=class_num, kernel_size=(1, 17), stride=(1, 2), padding=(0, 8))

    def forward(self, x):
        x = self.stn(x)

        out1 = self.stage1(x)  # 128x24x44
        out = self.down1(out1)  # 64x24x44
        out2 = self.stage2(out)  # 128x12x22
        out = self.down2(out2)  # 128x12x22
        out3 = self.stage3(out)  # 256x4x20
        out1 = self.outconv1(out1)  # 128x4x20
        out2 = self.outconv2(out2)  # 128x4x20
        logits = torch.cat((out1, out2, out3), 1)  # 512x4x20
        logits = self.container(logits)  # 512x2x18
        top, bottom = torch.split(logits, 1, dim=2)
        logits = torch.cat((top, bottom), 3)  # 512x1x36
        logits = self.container2(logits)  # cx1x18
        logits = logits.permute(0, 1, 3, 2)  # 18xcx1
        logits = torch.Tensor.squeeze(logits, dim=3)  # nx18xc

        return logits


CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新', '学', '港', '澳', '警', '使', '领', '应', '急', '挂',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '-'
        ]
