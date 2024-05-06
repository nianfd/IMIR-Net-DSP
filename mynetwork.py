import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1, conv3x3
import torch.nn.functional as F

class CA_Enhance(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA_Enhance, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes // 2, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, depth):
        x = torch.cat((rgb, depth), dim=1)
        #print(x.shape)
        max_pool_x = self.max_pool(x)
        #print(self.relu1(self.fc1(max_pool_x)).shape)

        max_out = self.fc2(self.relu1(self.fc1(max_pool_x)))
        out = max_out
        depth = depth.mul(self.sigmoid(out))
        return depth

class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class CA_SA_Enhance(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA_SA_Enhance, self).__init__()

        self.self_CA_Enhance = CA_Enhance(in_planes)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, rgb, depth):
        x_d = self.self_CA_Enhance(rgb, depth)
        sa = self.self_SA_Enhance(x_d)
        depth_enhance = depth.mul(sa)
        return depth_enhance

class DilatedEncoder(nn.Module):
    """
    Dilated Encoder for YOLOF.
    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
          which are 1x1 conv + 3x3 conv
        - the dilated residual block
    """

    def __init__(self,
                 in_channels=2048,
                 encoder_channels=512,
                 block_mid_channels=128,
                 #num_residual_blocks=4,
                 num_residual_blocks=3,
                 #block_dilations=[2, 4, 6, 8]
                 block_dilations = [2, 4, 6]
                 ):
        super(DilatedEncoder, self).__init__()
        # fmt: off
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = block_dilations

        assert len(self.block_dilations) == self.num_residual_blocks

        # init
        self._init_layers()
        self._init_weight()

    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(self.in_channels,
                                      self.encoder_channels,
                                      kernel_size=1)
        self.lateral_norm = nn.BatchNorm2d(self.encoder_channels)
        self.fpn_conv = nn.Conv2d(self.encoder_channels,
                                  self.encoder_channels,
                                  kernel_size=3,
                                  padding=1)
        self.fpn_norm = nn.BatchNorm2d(self.encoder_channels)
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                MyBottleneck(
                    self.encoder_channels,
                    self.block_mid_channels,
                    dilation=dilation
                )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def xavier_init(self, layer):
        if isinstance(layer, nn.Conv2d):
            # print(layer.weight.data.type())
            # m.weight.data.fill_(1.0)
            nn.init.xavier_uniform_(layer.weight, gain=1)

    def _init_weight(self):
        self.xavier_init(self.lateral_conv)
        self.xavier_init(self.fpn_conv)
        for m in [self.lateral_norm, self.fpn_norm]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out = self.lateral_norm(self.lateral_conv(feature))
        out = self.fpn_norm(self.fpn_conv(out))
        return self.dilated_encoder_blocks(out)


class MyBottleneck(nn.Module):

    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: int = 128,
                 dilation: int = 1):
        super(MyBottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out

class MyResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(MyResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)  这里将全连接层注释掉了
        #self.fc1 = nn.Linear(512, 256) #只用l4层的特征是512
        #self.fc1 = nn.Linear(512, 256) #
        # self.calorie = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        # self.mass = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        # self.fat = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        # self.carb = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        # self.protein = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        # self.encoder_l3 = DilatedEncoder(in_channels=1024)
        # self.encoder_l4 = DilatedEncoder(in_channels=2048)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        #print('11test')
        #print(block.expansion)
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        l3_fea = self.layer3(x)
        #l3_fea = self.encoder_l3(x)
        l4_fea = self.layer4(l3_fea)
        #x = self.encoder_l4(x)
        #print(x.shape)
        #x = self.avgpool(x)
        #print(x.shape)
        #l3_fea_pool = self.avgpool(l3_fea)
        #print(l3_fea_pool.shape)
        #x = l3_fea_pool + x

        #x = torch.flatten(x, 1)


        # x = self.fc1(x)
        # embedding = F.relu(x)
        # # embedding = F.dropout(embedding, self.training)
        # results = []
        # results.append(self.calorie(embedding).squeeze())
        # results.append(self.mass(embedding).squeeze())
        # results.append(self.fat(embedding).squeeze())
        # results.append(self.carb(embedding).squeeze())
        # results.append(self.protein(embedding).squeeze())
        # return results
        # x = self.fc(x) 这里注释掉了最后一个全连接层，直接输出提取的特征

        return l3_fea, l4_fea

    def forward(self, x):
        return self._forward_impl(x)

class MyResNetRGBD(nn.Module):
    def __init__(self):
        super(MyResNetRGBD, self).__init__()
        self.model_rgb = MyResNet(Bottleneck, [3, 4, 23, 3])  # 这里具体的参数参考库中源代码
        self.model_rgb.load_state_dict(torch.load('food2k_resnet101_0.0001.pth'), strict=False)

        self.model_depth = MyResNet(Bottleneck, [3, 4, 23, 3])  # 这里具体的参数参考库中源代码
        self.model_depth.load_state_dict(torch.load('food2k_resnet101_0.0001.pth'), strict=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc1 = nn.Linear(1024, 256) #
        self.fc1 = nn.Linear(512, 256)  #
        self.calorie = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.mass = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.fat = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.carb = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.protein = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))

        self.encoder_l3_rgb = DilatedEncoder(in_channels=1024)
        self.encoder_l4_rgb = DilatedEncoder(in_channels=2048)

        self.encoder_l3_depth = DilatedEncoder(in_channels=1024)
        self.encoder_l4_depth = DilatedEncoder(in_channels=2048)

        self.CA_SA_Enhance_3 = CA_SA_Enhance(2048)
        self.CA_SA_Enhance_4 = CA_SA_Enhance(4096)

        self.con2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.con2d_t = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.fc_t = nn.Linear(512,512)

    def forward(self, x_rgb, x_depth):
        x_fea = self.model_rgb.conv1(x_rgb)
        x_fea = self.model_rgb.bn1(x_fea)
        x_fea = self.model_rgb.relu(x_fea)
        x_fea = self.model_rgb.maxpool(x_fea)

        x_fea = self.model_rgb.layer1(x_fea)
        x_fea = self.model_rgb.layer2(x_fea)
        l3_fea_rgb = self.model_rgb.layer3(x_fea)

        ######################################
        x_fea_depth = self.model_depth.conv1(x_depth)
        x_fea_depth = self.model_depth.bn1(x_fea_depth)
        x_fea_depth = self.model_depth.relu(x_fea_depth)
        x_fea_depth = self.model_depth.maxpool(x_fea_depth)

        x_fea_depth = self.model_depth.layer1(x_fea_depth)
        x_fea_depth = self.model_depth.layer2(x_fea_depth)
        l3_fea_depth = self.model_depth.layer3(x_fea_depth)

        # rgb refine depth
        l3_fea_depth_rgb = self.CA_SA_Enhance_3(l3_fea_depth, l3_fea_rgb)
        l3_fea_depth = l3_fea_depth + l3_fea_depth_rgb

        l4_fea_depth = self.model_depth.layer4(l3_fea_depth)
        l4_fea_rgb = self.model_rgb.layer4(l3_fea_rgb)

        #depth refine rgb
        l4_fea_rgb_depth = self.CA_SA_Enhance_4(l4_fea_rgb, l4_fea_depth)
        l4_fea_rgb = l4_fea_rgb + l4_fea_rgb_depth


        l3_fea_rgb_dilate = self.encoder_l3_rgb(l3_fea_rgb)
        l4_fea_rgb_dilate = self.encoder_l4_rgb(l4_fea_rgb)

        # l3_fea_rgb_pool = self.avgpool(l3_fea_rgb_dilate)
        # l4_fea_rgb_pool = self.avgpool(l4_fea_rgb_dilate)
        l4_fea_rgb_dilate_up = torch.nn.functional.interpolate(l4_fea_rgb_dilate, scale_factor=2, mode='bilinear', align_corners=False)
        #rgb_fea = l3_fea_rgb_pool + l4_fea_rgb_pool_up
        rgb_fea_cat = torch.cat((l3_fea_rgb_dilate, l4_fea_rgb_dilate_up), dim=1)
        #print(rgb_fea.shape)
        rgb_fea = self.con2d(rgb_fea_cat)
        rgb_fea = self.relu(rgb_fea)
        rgb_fea = self.avgpool(rgb_fea)

        rgb_fea_t = self.con2d_t(rgb_fea_cat)
        rgb_fea_t = self.relu(rgb_fea_t)
        rgb_fea_t = self.avgpool(rgb_fea_t)
        rgb_fea_t = torch.squeeze(rgb_fea_t)
        rgb_fea_t = self.fc_t(rgb_fea_t)
        #rgb_fea_t /= rgb_fea_t.norm(dim=1, keepdim=True)
        #l3_fea = self.encoder_l3(x)
        #l4_fea = self.layer4(l3_fea)

        #l3_fea_rgb, l4_fea_rgb = self.model_rgb(x_rgb)
        #l3_fea_depth, l4_fea_depth = self.model_depth(x_depth)
        #print(l3_fea_rgb.shape)
        #print(l4_fea_rgb.shape)
        # l3_fea_depth_rgb = self.CA_SA_Enhance_3(l3_fea_depth, l3_fea_rgb)
        # l3_fea_depth = l3_fea_depth + l3_fea_depth_rgb
        #
        # l4_fea_rgb_depth = self.CA_SA_Enhance_4(l4_fea_rgb, l4_fea_depth)
        # l4_fea_rgb = l4_fea_rgb + l4_fea_rgb_depth
        #
        # l3_fea_rgb = self.encoder_l3_rgb(l3_fea_rgb)
        # l4_fea_rgb = self.encoder_l4_rgb(l4_fea_rgb)
        #
        # l3_fea_rgb_pool = self.avgpool(l3_fea_rgb)
        # l4_fea_rgb_pool = self.avgpool(l4_fea_rgb)
        # rgb_fea = l3_fea_rgb_pool + l4_fea_rgb_pool


        #l3_fea_depth = self.encoder_l3_depth(l3_fea_depth)
        #l4_fea_depth = self.encoder_l4_depth(l4_fea_depth)

        # l3_fea_depth_pool = self.avgpool(l3_fea_depth)
        # l4_fea_depth_pool = self.avgpool(l4_fea_depth)
        # depth_fea = l3_fea_depth_pool + l4_fea_depth_pool

        #rgbd_fea = rgb_fea + depth_fea
        #rgbd_fea = torch.cat((rgb_fea, depth_fea), dim=1)

        embedding = torch.flatten(rgb_fea, 1)
        embedding = self.fc1(embedding)
        embedding = F.relu(embedding)
        results = []
        results.append(self.calorie(embedding).squeeze())
        results.append(self.mass(embedding).squeeze())
        results.append(self.fat(embedding).squeeze())
        results.append(self.carb(embedding).squeeze())
        results.append(self.protein(embedding).squeeze())
        return results, rgb_fea_t





