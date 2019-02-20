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
testset=my_data(image_set='val',transform=image_transform,target_transform=mask_transform)
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
        self.conv1=vgg.features[:3] #64
        self.conv2=vgg.features[3:7] #128
        self.conv3=vgg.features[7:14] #256
        self.conv4=vgg.features[14:21] #512
        self.conv5=vgg.features[21:28] #512
        self.pool=vgg.features[-1]
        self.centre=deconv(512,512,256,True) #256
        self.up5=deconv(768,512,256,True)
        self.up4=deconv(768,512,128,True)
        self.up3=deconv(384,256,64,True)
        self.up2=deconv(192,128,32,True)
        self.up1=nn.Conv2d(32+64,1,3,padding=1)
    def forward(self, input):
        self.layer1=self.conv1(input)
        self.layer2=self.conv2(self.layer1)
        self.layer3=self.conv3(self.layer2)
        self.layer4=self.conv4(self.layer3)
        self.layer5=self.conv5(self.layer4)

        self.middle=self.centre(self.pool(self.layer5))
        # print (self.layer1.shape)
        # print (self.layer2.shape)
        # print (self.layer3.shape)
        # print (self.layer4.shape)
        # print (self.layer5.shape)


        self.middle=torch.nn.functional.pad(self.middle,(0,1,0,0),mode='replicate')
        # print (self.middle.shape)
        self.uplayer5 = self.up5 (torch.cat((self.middle,self.layer5),dim=1))
        self.uplayer4 = self.up4 (torch.cat((self.uplayer5,self.layer4),dim=1))
        self.uplayer3 = self.up3 (torch.cat((self.uplayer4,self.layer3),dim=1))
        self.uplayer2 = self.up2 (torch.cat((self.uplayer3,self.layer2),dim=1))
        self.uplayer1 = self.up1 (torch.cat((self.uplayer2,self.layer1),dim=1))
        # print (self.uplayer5.shape)
        # print (self.uplayer4.shape)
        # print (self.uplayer3.shape)
        # print (self.uplayer2.shape)
        # print (self.uplayer1.shape)


        return self.uplayer1

class Diceloss(nn.Module):
    def __init__(self):
        super(Diceloss,self).__init__()
    def dice_coef(self,x,y):
        numerator=2*torch.sum(x*y)+0.0001
        denominator=torch.sum(x**2)+torch.sum(y**2)+0.0001
        return numerator/denominator
    def forward(self, x,y):
        return 1-self.dice_coef(x,y)
class Bce_Diceloss(nn.Module):
    def __init__(self,bce_rate=0.5):
        super(Bce_Diceloss,self).__init__()
        self.rate=bce_rate
    def dice_coef(self,x,y):
        numerator=2*torch.sum(x*y)+0.0001
        denominator=torch.sum(x**2)+torch.sum(y**2)+0.0001
        return numerator/denominator
    def forward(self, x,y):
        return (1-self.dice_coef(x,y))*(1-self.rate)+self.rate*torch.nn.functional.binary_cross_entropy(x,y)
def train(epoch):
    model=UNET()
    model.train()
    model.to(device)
    criterion=Diceloss()
    optimize=torch.optim.Adam(model.parameters(),lr=0.001)
    for i in range(epoch):
        tmp=0
        for image,mask in trainload:
            image,mask=image.to(device,dtype=dtype),mask.to(device,dtype=dtype)
            optimize.zero_grad()
            pred=model(image)
            loss=criterion(pred,mask)
            loss.backward()
            optimize.step()
            tmp=loss.data
            # print ("loss ",tmp)
            break
        print ("{0} epoch ,loss is {1}".format(i,tmp))
    return model



model=train(1)

# https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
# https://github.com/ybabakhin/kaggle_salt_bes_phalanx/blob/master/bes/losses.py
# https://arxiv.org/pdf/1606.04797.pdf#pdfjs.action=download
# okular