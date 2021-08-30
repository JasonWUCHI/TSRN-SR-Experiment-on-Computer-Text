import os
import numpy as np
import torch
from torchvision.transforms import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
  def __init__(self , img_dir , transform = None):
    self.img_dir = img_dir
    self.transform = transform
  
  def __len__(self):
    return len(os.listdir(self.img_dir))

  def __getitem__(self , idx):
    img_path = os.path.join(self.img_dir , "screenshot_%05d.png" % idx)
    img = Image.open(img_path).convert('RGB')
    if self.transform:
      img = self.transform(img)
    return img

class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)
        return img_tensor

class alignCollate_syn(object):
    def __init__(self, imgH=64, imgW=256, down_sample_scale=4, keep_ratio=False, min_ratio=1, mask=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask

    def __call__(self, batch):
        images = batch
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)

        images_hr = [transform(image) for image in images]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        images_lr = [image.resize((image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale), Image.BICUBIC) for image in images]
        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        return images_hr, images_lr