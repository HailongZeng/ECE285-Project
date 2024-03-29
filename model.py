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
def weights_init(m):
    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Initialize the MSELoss function
criterion1 = nn.MSELoss()
# Initialize the L1Loss function
criterion2 = nn.L1Loss()    
real_label = Variable(torch.cuda.FloatTensor(1).fill_(1.0), requires_grad = False)
fake_label = Variable(torch.cuda.FloatTensor(1).fill_(0.0), requires_grad = False)
def cal_loss_Gan(D, real, fake):
    '''
    input:
        D--Discriminator
        real--X from X domain  or Y from Y domain 
        fake--F(Y) generated by using Y from Y domain or G(X) generated by using X from X domain
    '''
    pred_real = D(real)
    pred_fake = D(fake.detach())
    loss_D_real = criterion1(pred_real, real_label)
    loss_D_fake = criterion1(pred_fake, fake_label)
    loss_D = 0.5 * (loss_D_fake + loss_D_real) 
    return loss_D

def cal_loss_Cycle(net, real, fake):
    '''
    input:
        net:
            G--Generator which generate image from X domain to Y domain
            or F--Generator which generate image from Y domain to X domain
        real--X from X domain  or Y from Y domain 
        fake--F(Y) generated by using Y from Y domain or G(X) generated by using X from X domain
    return: Cycle loss
    '''
    loss_Cycle = criterion2(real, net(fake))
    return loss_Cycle