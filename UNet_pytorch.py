#!/usr/bin/env python
# coding: utf-8

# In[24]:


import torch
import cv2 as cv
import os
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import datasets, transforms

#import unet_model


# # 1.定义一些超参数

# In[25]:


BATCH_SIZE = 10
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # 2.载入数据集

# ## 2.1.训练集

# In[32]:


class utcntrain(Dataset):#数据集

    def __init__(self, file_dir):
        """实现初始化方法，在初始化的时候将数据读载入"""
        for files in os.walk(file_dir):  #获取当前路径下所有文件名   
            files
        self.df=files[2]#一个索引，所有训练及文件的名称组成的一个list，os.walk()返回的[2]才是文件名。
        
    def __len__(self):# 返回df的长度
      
        return len(self.df)
    
    def __getitem__(self, idx):#idx是索引，根据这个索引返回一个样本

        return cv.imread('images/'+str(self.df[idx])),cv.imread('labels/'+str(self.df[idx])) #读一张图为numpy


# DataLoader为我们提供了对Dataset的读取操作，常用参数有：batch_size(每个batch的大小)、 shuffle(是否进行shuffle操作)、 num_workers(加载数据的时候使用几个子进程)。下面做一个简单的操作

# In[33]:


'''测试一下'''
traindata = utcntrain('images')#参数为路径
train_loader = torch.utils.data.DataLoader(traindata, batch_size=10, shuffle=False, num_workers=0)


# ## 2.2.测试集(暂不使用）
class utcnlab(Dataset):#数据集

    def __init__(self, file_dir):
        """实现初始化方法，在初始化的时候将数据读载入"""
        for files in os.walk(file_dir):  #获取当前路径下所有文件名   
            files
        self.df=files[2]#一个索引，所有训练及文件的名称组成的一个list，os.walk()返回的[2]才是文件名。
        
    def __len__(self):# 返回df的长度
      
        return len(self.df)
    
    def __getitem__(self, idx):#idx是索引，根据这个索引返回一个样本

        return cv.imread('images/'+str(self.df[idx])) #读一张图为numpy
# # 3.Model

# ## 3.1 部分

# In[34]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ## 3.2 整合

# In[35]:


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# # 4.训练

# ## 4.1.训练函数

# In[36]:


model = UNet(3,1).to(DEVICE)
optimizer = optim.Adam(model.parameters())


# In[37]:


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# ## 4.2. 开始训练

# In[38]:


for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)

torch.save(net.state_dict(),unet.pt)


# In[ ]:




