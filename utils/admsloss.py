import torch
import torch.nn as nn
import torch.nn.functional as F

class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        # x = F.normalize(x, dim=1)
        # wf = self.fc(x)
        wf = x
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        # return -torch.mean(L)
        return -L


if __name__ == "__main__":
    ## 验证直接点选(index slicing操作)结果能否使计算的结果变得正确
    in_features = 10
    out_features = 5
    criterion = AdMSoftmaxLoss(in_features, out_features, s=1, m=0.5)
    x = torch.rand((5, 5))
    x = F.normalize(x, dim=1)
    print(x)
    label = torch.Tensor([0, 2, 3, 1, 1]).long()
    loss = criterion(x, label)
    x_margin = x.clone()
    print(x_margin)
    x_margin[label>=0,label] -= 0.5
    x_margin /= 1
    print(x_margin)
    # loss2 = - (x_margin[label>=0, label] - torch.log(torch.sum(torch.exp(x_margin), dim=1)))
    loss2 = F.cross_entropy(x_margin, label, reduction='none')
    print(loss)
    print(loss2)