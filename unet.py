import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torchvision as tv
import random
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader as dataloader
import nonechucks as nc
from voc_seg import my_data,label_acc_score,voc_colormap,seg_target
vgg=tv.models.vgg11_bn(pretrained=True)
image_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
mask_transform=transforms.Compose([transforms.ToTensor()])
trainset=my_data(transform=image_transform,target_transform=mask_transform)
testset=my_data('val',image_transform,mask_transform)
trainload=torch.utils.data.DataLoader(trainset,batch_size=4)
testload=torch.utils.data.DataLoader(testset,batch_size=4)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')
dtype=torch.float32
class deconv(nn.Module):
    def __init__(self,inchannel,middlechannel,outchannel,transpose=False):
        super(deconv,self).__init__()
        if transpose:
            self.block=nn.Sequential(nn.Conv2d(inchannel,middlechannel,3,padding=1),
                                   nn.BatchNorm2d(middlechannel),
                                   nn.ReLU(inplace=True),
                                   # nn.Conv2d(middlechannel,middlechannel,3,padding=1),
                                   # nn.BatchNorm2d(middlechannel),
                                   # nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(middlechannel,outchannel,3,2,1,1) # use out_pading to minus one of padding
                                     )
        else:
            self.block=nn.Sequential(nn.Conv2d(inchannel,middlechannel,3,padding=0),
                                   nn.BatchNorm2d(middlechannel),
                                   nn.ReLU(inplace=True),
                                   # nn.Conv2d(middlechannel,middlechannel,3,padding=0),
                                   # nn.BatchNorm2d(middlechannel),
                                   # nn.ReLU(inplace=True),
                                   nn.Conv2d(middlechannel,outchannel,1),         #since unsampling cann't change the channel num ,have to change channel num before next block
                                   nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True) # transpose is upsample and conv, now try conv and upsample need to confirm
                                     )
    def forward(self, input):
        return self.block(input)
class UNET(nn.Module):
    def __init__(self):
        super(UNET,self).__init__()
        self.conv1=vgg.features[:4] #64
        self.conv2=vgg.features[4:8] #128
        self.conv3=vgg.features[8:15] #256
        self.conv4=vgg.features[15:22] #512
        self.conv5=vgg.features[22:] #512
        self.centre=deconv(512,512,256,True) #256
        self.up5=deconv(768,512,256,True)
        self.up4=deconv(768,512,256,True)
        self.up3=deconv(512,256,128,True)
        self.up2=deconv(256,128,64,True)
        self.up1=deconv(128,64,1,True)
    def forward(self, input):
        self.layer1=self.conv1(input)
        self.layer2=self.conv2(self.layer1)
        self.layer3=self.conv3(self.layer2)
        self.layer4=self.conv4(self.layer3)
        self.layer5=self.conv5(self.layer4)

        self.middle=self.centre(self.layer5)

        self.uplayer5 = self.up5 (torch.cat((self.middle,self.layer5),dim=1))
        self.uplayer4 = self.up4 (torch.cat((self.uplayer5,self.layer4),dim=1))
        self.uplayer3 = self.up3 (torch.cat((self.uplayer4,self.layer3),dim=1))
        self.uplayer2 = self.up2 (torch.cat((self.uplayer3,self.layer2),dim=1))
        self.uplayer1 = self.up1 (torch.cat((self.uplayer2,self.layer1),dim=1))
        return self.uplayer1

def train(epoch):
    model=UNET()
    model.train()
    model.to(device)
    criterion=nn.CrossEntropyLoss()

    optimize=torch.optim.Adam(model.parameters(),lr=0.001)
    for i in range(epoch):
        tmp=0
        for image,mask in trainload:
            image,mask=image.to(device,dtype=dtype),mask.to(device,dtype=torch.long)
            optimize.zero_grad()
            pred=model(image)
            loss=criterion(pred,mask)
            loss.backward()
            optimize.step()
            tmp=loss.data
            # print ("loss ",tmp)
        print ("{0} epoch ,loss is {1}".format(i,tmp))
    return model
https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
https://github.com/ybabakhin/kaggle_salt_bes_phalanx/blob/master/bes/losses.py
https://arxiv.org/pdf/1606.04797.pdf#pdfjs.action=download
okular