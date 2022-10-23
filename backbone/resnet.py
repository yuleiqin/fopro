import torch
import torch.nn as nn
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Function


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super(ScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        print(self.scale)
        return input * self.scale


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class ResNet(nn.Module):

#     def __init__(self, block, layers, low_dim=128, in_channel=3, width=1, num_class=1000, use_norm=False):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         ## Feature backbone
#         self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.base = int(64 * width)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, self.base, layers[0])
#         self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         ## Classifier
#         self.classifier = nn.Linear(self.base * 8 * block.expansion, num_class)
#         self.l2norm = Normalize(2)
#         ## If use normalization on extracted features
#         ## the classifier is actually cosine classifier by prototype
#         self.use_norm = use_norm
#         self.low_dim = low_dim
#         if self.low_dim != -1:
#             ## projection MLP
#             self.fc1 = nn.Linear(self.base * 8 * block.expansion, 2048)
#             self.fc2 = nn.Linear(2048, low_dim)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         feat = x.view(x.size(0), -1)
#         ## optional normalization
#         if self.use_norm:
#             feat = self.l2norm(feat)
#         ## classifier
#         out = self.classifier(feat)  
#         ## optional mapping to low-dimension
#         if self.low_dim != -1:
#             feat = F.relu(self.fc1(feat))
#             feat = self.fc2(feat)
#             feat = self.l2norm(feat) 
#         return out, feat

    
class ResNet_Encoder(nn.Module):
    ## in_channel = 512 for Res18 and Res34
    ## in_channel = 2048 for Res50 and Res101
    def __init__(self, block, layers, in_channel=3, width=1):
        self.inplanes = 64
        super(ResNet_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.base = int(64 * width)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)
        self.num_out_channel = self.base * 8 * block.expansion
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        return feat

    
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Encoder(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        ## 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict}
        ## 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        ## 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Encoder(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        ## 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict}
        ## 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        ## 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Encoder(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        ## 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict}
        ## 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        ## 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Encoder(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        ## 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict}
        ## 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        ## 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":
    inputs = torch.rand((5, 3, 224, 224))
    # inputs = torch.rand((5, 3, 448, 448))

    model = resnet18(False, width=1)
    print("res18", model.num_out_channel)
    outputs = model(inputs)
    print(outputs.size())
    
    model = resnet34(False, width=1)
    print("res34", model.num_out_channel)
    outputs = model(inputs)
    print(outputs.size())

    model = resnet101(False, width=1)
    print("res101", model.num_out_channel)
    outputs = model(inputs)
    print(outputs.size())

    model = resnet50(False, width=1)
    print("res50", model.num_out_channel)
    outputs = model(inputs)
    print(outputs.size())
    
# res18 512
# torch.Size([5, 512])
# res34 512
# torch.Size([5, 512])
# res101 2048
# torch.Size([5, 2048])
# res50 2048
# torch.Size([5, 2048])



