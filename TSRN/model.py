import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed

class GruBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(GruBlock, self).__init__()
    assert out_channels % 2 == 0
    self.conv1 = nn.Conv2d(in_channels , out_channels , kernel_size=1 , padding = 0)
    self.gru = nn.GRU(out_channels , out_channels // 2 , bidirectional = True , batch_first = True)
  
  def forward(self , x):
    x = self.conv1(x)
    x = x.permute(0 , 2 , 3 ,1).contiguous()
    b = x.size()
    x = x.view(b[0] * b[1], b[2], b[3])
    x , _ = self.gru(x)
    x = x.view(b[0] , b[1] , b[2] , b[3])
    x = x.permute(0 , 3 , 1 , 2)
    return x

class mish(nn.Module):
  def __init__(self,):
    super(mish , self).__init__()
    self.activated = True

  def forward(self , x):
    if self.activated:
      x = x * (torch.tanh(F.softplus(x)))
    return x

class RecurrentResidualBlock(nn.Module):
  def __init__(self, channels):
    super(RecurrentResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size = 3 , padding = 1)
    self.bn1 = nn.BatchNorm2d(channels)
    self.gru1 = GruBlock(channels , channels)
    self.prelu = mish()
    self.conv2 = nn.Conv2d(channels , channels , kernel_size=3 , padding=1)
    self.bn2 = nn.BatchNorm2d(channels)
    self.gru2 = GruBlock(channels , channels)
  
  def forward(self , x):
    residual = self.conv1(x)
    residual = self.bn1(residual)
    residual = self.prelu(residual)
    residual = self.conv2(residual)
    residual = self.bn2(residual)
    residual = self.gru1(residual.transpose(-1 , -2)).transpose(-1,-2)

    return self.gru2(x+residual)

class UpsampleBlock(nn.Module):
  def __init__(self, in_channels , upscale):
    super(UpsampleBlock , self).__init__()
    self.conv = nn.Conv2d(in_channels , in_channels * upscale ** 2 , kernel_size=3 , padding=1)
    self.pixel_shuffle = nn.PixelShuffle(upscale)
    self.prelu = mish()

  def forward(self , x):
    x = self.conv(x)
    x = self.pixel_shuffle(x)
    x = self.prelu(x)
    return x

class TSRN(nn.Module):
  def __init__(self , scale_factor=4 , width=128 , height=32 , STN=False , srb_nums=5, mask=True, hidden_units=32 ):
    super(TSRN, self).__init__()
    in_planes = 3
    if mask:
      in_planes = 4
    assert math.log(scale_factor,2)%1 == 0
    upsample_block_num = int(math.log(scale_factor,2))
    self.block1 = nn.Sequential(
        nn.Conv2d(in_planes , 2* hidden_units , kernel_size=9 , padding = 4),
        nn.PReLU()
    )
    self.srb_nums = srb_nums
    for i in range(srb_nums):
      setattr(self , 'block%d' % (i+2), RecurrentResidualBlock(2*hidden_units))
    
    setattr(self , 'block%d' % (srb_nums+2),
            nn.Sequential(
                nn.Conv2d(2*hidden_units , 2*hidden_units , kernel_size = 3 , padding=1),
                nn.BatchNorm2d(2*hidden_units)
            ))
    block_ = [UpsampleBlock(2*hidden_units , 2) for _ in range(upsample_block_num)]
    block_.append(nn.Conv2d(2*hidden_units , in_planes ,  kernel_size=9 , padding = 4))
    setattr(self, 'block%d' % (srb_nums+3) , nn.Sequential(*block_))

  def forward(self , x):
    block = {'1': self.block1(x)}
    for i in range(self.srb_nums + 1):
      block[str(i+2)] = getattr(self, 'block%d' % (i+2))(block[str(i+1)])

    block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3))((block['1'] + block[str(self.srb_nums + 2)]))
    output = torch.tanh(block[str(self.srb_nums + 3)])
    return output