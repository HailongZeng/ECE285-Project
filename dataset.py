import torchvision.utils as vutils
import copy
import math
import os
import numpy as np
from PIL import Image, ImageFile
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as td
import torchvision as tv
import pandas as pd
from torch.autograd import Variable
from io import BytesIO
import itertools
from image_pool import ImagePool
import time

class LandscapeDataset(td.Dataset):
    def __init__(self, root_dir, category = "mountain", image_size = (224, 224)):
        super(LandscapeDataset, self).__init__()
        self.image_size = image_size
        self.category = category
        json_path = os.path.join(root_dir, "licenses/{}_photos_info.json".format(category))
        self.data = pd.read_json(json_path)  # read the json file 
        self.data = self.data.loc[:,self.data.iloc[2] != '3']   # drop the license which is '3'
        self.data = self.data.loc[:,self.data.iloc[2] != '6']  # drop the license which is '6'
        self.data = self.data.T   # transpose the DataFrame

        self.images_dir = os.path.join(root_dir, self.category)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "LandscapeDataset(category={}, image_size={})". format(self.category, self.image_size)
    
    def __getitem__(self,idx):
        valid=False
        i = 0
        while not valid:
            try:
                img_path = os.path.join(self.images_dir, self.data.iloc[idx][0]) 
                img = Image.open(img_path).convert('RGB')
                valid = True
            except FileNotFoundError:
                idx = i
                i += 1
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        x = transform(img)          # transformed image
        d = self.data.iloc[idx][2]   # license of the image
        return x
class ArtDataset(td.Dataset):
    def __init__(self, art_root_dir, mode="train", image_size=(224, 224)):
        super(ArtDataset, self).__init__()
        self.image_size = image_size
        self.mode = mode
        self.data = pd.read_csv(os.path.join(art_root_dir, "Style/style_%s.csv" % mode))
        self.data = self.data.loc[self.data.iloc[:,1] == 4]   # vincent-van-gogh
        self.images_dir = os.path.join(art_root_dir,'wikiart')

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "ArtDataset(mode={}, image_size={})". format(self.mode, self.image_size)
    
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.images_dir, self.data.iloc[idx][0])
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(img_path).convert('RGB')
        except UnicodeEncodeError:
            img_path = os.path.join(self.images_dir, self.data.iloc[idx-1][0]) 
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(img_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        x = transform(img)
        d = self.data.iloc[idx][1]
        return x
    
def myimshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h