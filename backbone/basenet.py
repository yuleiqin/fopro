from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
import math


class AlexNet_Encoder(nn.Module):
    ## in_channel = 4096
    def __init__(self, pretrained=False):
        super(AlexNet_Encoder, self).__init__()
        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential()
        ## only keep until penultimate layer ->4096->4096
        ## delete the mapping from 4096->1000 (num_class)
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.num_out_channel = model_alexnet.classifier[-1].in_features
        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        feat = self.classifier(x)
        return feat


class VGG_Encoder(nn.Module):
    ## in_channel = 4096
    def __init__(self, pretrained=False):
        super(VGG_Encoder, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        ## only keep until penultimate layer ->4096->4096
        ## delete the mapping from 4096->1000 (num_class)
        classifier = list(vgg16.classifier._modules.values())
        self.num_out_channel = classifier[-1].in_features
        self.classifier = nn.Sequential(*classifier[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        feat = self.classifier(x)
        return feat


class BCNN_encoder(nn.Module):
    """
    BCNN

    The structure of BCNN is as follows:
        conv1_1 (64) -> relu -> conv1_2 (64) -> relu -> pool1(64*224*224)
    ->  conv2_1(128) -> relu -> conv2_2(128) -> relu -> pool2(128*112*112)
    ->  conv3_1(256) -> relu -> conv3_2(256) -> relu -> conv3_3(256) -> relu -> pool3(256*56*56)
    ->  conv4_1(512) -> relu -> conv4_2(512) -> relu -> conv4_3(512) -> relu -> pool4(512*28*28)
    ->  conv5_1(512) -> relu -> conv5_2(512) -> relu -> conv5_3(512) -> relu(512*28*28)
    ->  bilinear pooling(512**2)
    ->  fc(n_classes)

    The network input 3 * 448 * 448 image
    The output of last convolution layer is 512 * 14 * 14

    Extends:
        torch.nn.Module
    """
    def __init__(self, pretrained=False, num_out_channel=512**2):
        super().__init__()
        self._pretrained = pretrained
        vgg16 = models.vgg16(pretrained=self._pretrained)
        self.features = nn.Sequential(*list(vgg16.features.children())[:-1])
        self.num_out_channel = num_out_channel
        if self.num_out_channel != 512**2:
            self.fc = nn.Linear(512**2, self.num_out_channel)
            if self._pretrained:
                # Init the fc layer
                nn.init.kaiming_normal_(self.fc.weight.data)
                if self.fc.bias is not None:
                    nn.init.constant_(self.fc.bias.data, val=0)
        else:
            self.fc = nn.Identity()
        # Freeze all layer in self.feature
        # for params in self.features.parameters():
        #   params.requires_grad = False

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        """
        Forward pass of the network

        Arguments:
            x [torch.Tensor] -- shape is (N, 3, 448, 448)

        Return:
            x [torch.Tensor] -- shape is (N, 200)
        """
        x = self.features(x)
        bp_output = self.bilinear_pool(x)
        feat = self.fc(bp_output)
        return feat

    @staticmethod
    def bilinear_pool(x):
        N, ch, h, w = x.shape
        # assert x.size() == (N, 512, 28, 28)
        x = x.view(N, 512, h*w)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (h * w)
        x = x.view(N, 512**2)
        x = torch.sqrt(x + 1e-5)
        x = F.normalize(x)
        assert x.size() == (N, 512**2)
        return x


if __name__ == "__main__":
    model = AlexNet_Encoder(pretrained=False)
    print("alex-net out channel {}".format(model.num_out_channel))
    # inputs = torch.rand((2, 3, 448, 448))
    inputs = torch.rand((2, 3, 224, 224))
    outputs = model(inputs)
    print(outputs.size())

    model = VGG_Encoder(pretrained=False)
    print("vgg out channel {}".format(model.num_out_channel))
    # inputs = torch.rand((2, 3, 448, 448))
    inputs = torch.rand((2, 3, 224, 224))
    outputs = model(inputs)
    print(outputs.size())

    model = BCNN_encoder(pretrained=True, num_out_channel=2048)
    print("bcnn out channel {}".format(model.num_out_channel))
    # inputs = torch.rand((2, 3, 448, 448))
    inputs = torch.rand((2, 3, 224, 224))
    outputs = model(inputs)
    print(outputs.size())
