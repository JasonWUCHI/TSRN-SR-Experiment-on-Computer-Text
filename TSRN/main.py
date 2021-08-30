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
BatchSize = 16
train_path = '/content/content/train_jpeg'
eval_path = '/content/content/eval_jpeg'
image_Height = 256
image_Width = 256
downSample = 4

#Model
srb = 5
hd_u = 32

#Loss
grd = True #gradient

#Optimizer
learning_rate = 0.001
epochs = 10
beta1 = 0.5

#loss tracking
training_loss = []
eval_loss = []

### check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### load dataset and data_loader
train_dataset = MyDataset(train_path)

train_loader = torch.utils.data.DataLoader(
    train_dataset , batch_size = BatchSize , shuffle = True ,
    collate_fn = alignCollate_syn(imgH=image_Height,imgW=image_Width, down_sample_scale=downSample, mask = True) , 
    drop_last = True
)

def eval(model):
    eval_dataset = MyDataset(eval_path)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size = BatchSize, shuffle = True, collate_fn = alignCollate_syn(imgH = image_Height, imgW= image_Width, down_sample_scale=downSample, mask = True), drop_last = True
    )

    image_crit = ImageLoss(gradient=grd, loss_weight=[1 , 1e-4])
    image_crit = image_crit.to(device)

    loss_list = []

    model.eval()
    for j, data in (enumerate(eval_loader)):
        for p in model.parameters():
            p.requires_grad = False

    images_hr , images_lr = data
    images_lr = images_lr.to(device)
    images_hr = images_hr.to(device)

    images_sr = model(images_lr)
    loss_im_eval = image_crit(images_sr,images_hr).mean() * 100
    loss_list.append(loss_im_eval.item())
    torch.cuda.empty_cache()

    eval_loss.append(sum(loss_list)/len(loss_list))

def train():
    train_dataset = MyDataset(train_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset , batch_size = BatchSize , shuffle = True , collate_fn = alignCollate_syn(imgH = image_Height, imgW = image_Width, down_sample_scale=downSample, mask = True) , drop_last = True
    )

    model = TSRN(scale_factor=downSample , width=image_Width , height=image_Height , STN=False , srb_nums=srb , mask=True , hidden_units=hd_u)
    model = model.to(device)
    image_crit = ImageLoss(gradient=grd, loss_weight=[1 , 1e-4])
    image_crit = image_crit.to(device)
    optimizer_G = torch.optim.Adam(model.parameters() , lr=learning_rate , betas=(beta1,0.999))

    #training_loss.clear()
    #eval_loss.clear()

    for epoch in range(epochs):
        print("Number of epoch:" , epoch)
        for j, data in (enumerate(train_loader)):
            model.train() #verify that model is training instead of evaluating

            for p in model.parameters():
                p.requires_grad = True
            iters = len(train_loader) * epoch + j + 1

            images_hr , images_lr = data

            images_lr = images_lr.to(device)
            images_hr = images_hr.to(device)

            image_sr = model(images_lr)
            loss_im = image_crit(image_sr , images_hr).mean() * 100

            optimizer_G.zero_grad()
            loss_im.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer_G.step()

            torch.cuda.empty_cache()

            if j % 33 == 0:
                print("Loss: ", loss_im)
                training_loss.append(loss_im.item())
                eval(model)

    torch.save(model.state_dict(), 'TextZoom.pkl')

torch.cuda.empty_cache()
train()

#evaluation
import matplotlib.pyplot as plt
plt.plot(np.linspace(1, len(training_loss) , len(training_loss))/5, training_loss)
plt.plot(np.linspace(1, len(eval_loss) , len(eval_loss))/5, eval_loss)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.locator_params(axis='x', nbins=20)
plt.legend(['Training Loss', 'Validation Loss'])
plt.savefig('demo_folder/validation.png')
plt.show()
