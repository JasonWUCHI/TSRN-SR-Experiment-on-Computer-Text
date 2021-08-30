import torch
import sys
import time
import os
from time import gmtime, strftime
from datetime import datetime
import math
import copy
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
import numpy as np

from dataset import MyDataset, alignCollate_syn
from loss import ImageLoss, GradientPriorLoss
from model import GruBlock, mish, RecurrentResidualBlock, UpsampleBlock, TSRN

###########   Variable Settings
#settings for the dataset
image_Height = 256
image_Width = 256
downSample = 4

#Model
srb = 5
hd_u = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########### Demo
model_eval = TSRN(scale_factor=downSample , width=image_Width , height=image_Height , STN=False , srb_nums=srb , mask=True , hidden_units=hd_u)
model_eval.load_state_dict(torch.load('demo_folder/TextZoom.pkl'))
model_eval = model_eval.to(device)
eval_img = Image.open("demo_folder/hr.png")
eval_img = eval_img.resize((image_Height // downSample, image_Width // downSample) , Image.BICUBIC)
img_tensor = transforms.ToTensor()(eval_img)

#add mask
mask = eval_img.convert('L')
thres = np.array(mask).mean()
mask = mask.point(lambda x: 0 if x > thres else 255)
mask = transforms.ToTensor()(mask)
img_tensor = torch.cat((img_tensor, mask), 0)

img_lr = img_tensor.unsqueeze(0)

img_lr = img_lr.to(device)
img_sr = model_eval(img_lr)

# tensor to ndarray
img_save = img_sr[0]*255
img_save = np.transpose(img_save.cpu().detach().numpy() , (1,2,0))
img_save = img_save[: , : , 0:3]
img_save = np.array(img_save, dtype=np.uint8)

if np.ndim(img_save)>3:
    assert img_save.shape[0] == 1
    img_save = img_save[0]
  
img_save = Image.fromarray(img_save , "RGB")
img_save.save('demo_folder/output.png')

blur_img = eval_img.resize((image_Height, image_Width), Image.BICUBIC)
blur_img.save('demo_folder/bicubic.png')
eval_img.save('demo_folder/small.png')