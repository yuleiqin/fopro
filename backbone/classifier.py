from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class MLP_classifier(nn.Module):
    def __init__(self, num_class=1000, in_channel=4096, num_hidden=-1, use_norm=False, use_sigmoid=False):
        super(MLP_classifier, self).__init__()
        self.num_hidden = num_hidden
        self.num_class = num_class
        self.use_norm = use_norm
        if num_hidden != -1:
            self.fc1 = nn.Linear(in_channel, num_hidden, bias=False)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(num_hidden, num_class, bias=False)
        else:
            self.fc = nn.Linear(in_channel, num_class, bias=False)
        self.l2norm = Normalize(2)
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def __norm_weight(self):
        if self.num_hidden != -1:
            for W in self.fc1.parameters():
                W = F.normalize(W, dim=1)
            for W in self.fc2.parameters():
                W = F.normalize(W, dim=1)
        else:
            for W in self.fc.parameters():
                ## W: [N_dim_in_channel, N_class]
                W = F.normalize(W, dim=1)
        return

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        if self.use_norm:
            ## 对输入特征首先进行归一化
            x = self.l2norm(x)
            ## 然后对权重本身进行归一化
            self.__norm_weight()
        if self.num_hidden != -1:
            x_out = self.fc2(self.relu(self.fc1(x)))
        else:
            x_out = self.fc(x)
        if self.use_sigmoid:
            x_out = self.sigmoid(x_out)
        return x_out


class Discriminator(nn.Module):
    def __init__(self, in_channel=4096):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential([
            nn.Linear(in_channel, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ])

    def forward(self, x, reverse=True, eta=1.0):
        if reverse:
            x = grad_reverse(x, eta)
        x = self.fc(x)
        x_out = F.sigmoid(x)
        return x_out


