import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np



def inverse_gumbel_cdf(y, mu, beta):
    return mu - beta * torch.log(-torch.log(y))

def gumbel_softmax_sampling(h, mu=0, beta=1, tau=0.1):
    """
    h : (N x K) tensor. Assume we need to sample a NxK tensor, each row is an independent r.v.
    """
    shape_h = h.shape
    p = F.softmax(h, dim=1)
    y = torch.rand(shape_h) + 1e-25  # ensure all y is positive.
   
    g = inverse_gumbel_cdf(y, mu, beta)
  
    # print('p.device',p.device)
    # print('g.device',g.device)
    g = g.cuda()
    x = torch.log(p) + g  # samples follow Gumbel distribution.
    # using softmax to generate one_hot vector:
   
    x = x/tau
   
    x = F.softmax(x, dim=1)  # now, the x approximates a one_hot vector.
    return x



class Route(nn.Module):
    def __init__(self, class_num = 4):
        super(Route, self).__init__()
        self.class_num = class_num
        self.conv1 = nn.Sequential(
            nn.Conv2d(		# input: 2,160,192
                in_channels=2,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.AvgPool2d(2),  # 16,160,192 --> 16,80,96
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.AvgPool2d(2),  # 8,80,96 --> 8,40,48
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Conv2d(
                in_channels=8,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.AvgPool2d(2),  # 1,40,48 --> 1,20,24
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        self.line = nn.Sequential(
            nn.Linear(
                in_features=20 * 24,
                out_features=self.class_num
            ),
            nn.Softmax()
        )
        

    # def gumbel_softmax_sampling(h, mu=0, beta=1, tau=0.1):
    #     """
    #     h : (N x K) tensor. Assume we need to sample a NxK tensor, each row is an independent r.v.
    #     """
    #     print('type(h)',type(h))
    #     shape_h = h.shape
    #     p = F.softmax(h, dim=1)
    #     y = torch.rand(shape_h) + 1e-25  # ensure all y is positive.
    #     g = mu - beta * np.log(-np.log(y))
    #     x = torch.log(p) + g  # samples follow Gumbel distribution.
    #     # using softmax to generate one_hot vector:
    #     x = x/tau
    #     x = F.softmax(x, dim=1)  # now, the x approximates a one_hot vector.
    #     return x

    
    def forward(self, source, target):
        
        x = torch.cat([source, target], dim=1)
        x = self.conv1(x)
        x = x.view(-1, 20 * 24)
        y = self.line(x)
        # y = gumbel_softmax_sampling(y)
        
        return y
