import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torchvision as tv
import random
import math
from functools import reduce
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader as dataloader
# import nonechucks as nc
from voc_seg import my_data,label_acc_score,voc_colormap,seg_target
vgg=tv.models.vgg11_bn(pretrained=True)
image_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
mask_transform=transforms.Compose([transforms.ToTensor()])
trainset=my_data(transform=image_transform,target_transform=mask_transform)
testset=my_data(image_set='test',transform=image_transform,target_transform=mask_transform)
trainload=torch.utils.data.DataLoader(trainset,batch_size=2)
testload=torch.utils.data.DataLoader(testset,batch_size=1)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print (device)
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
                                   # nn.ConvTranspose2d(middlechannel,outchannel,3,2), # use out_pading to minus one of padding
                                    nn.ConvTranspose2d(middlechannel,outchannel,3,2,1,1) # use out_pading to minus one of padding

                                     )
        else:
            self.block=nn.Sequential(nn.Conv2d(inchannel,middlechannel,3,padding=1),
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
class up(nn.Module):
    def __init__(self,inchannel_low,inchannel_same,middlechannel,outchannel,transpose=False):
        super(up,self).__init__()
        if  transpose:
            self.block=nn.ConvTranspose2d(inchannel_low,middlechannel,3,2,1,1)
            self.conv=nn.Sequential(nn.Conv2d(middlechannel+inchannel_same,outchannel,3,padding=1),
                                nn.BatchNorm2d(outchannel),
                                nn.ReLU(inplace=True),
                                # nn.ConvTranspose2d(middlechannel,outchannel,3,2,1,1)
                                )
        else:
            self.block=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                                     nn.Conv2d(inchannel_low,middlechannel,3,padding=1),
                                     nn.BatchNorm2d(middlechannel),
                                     nn.ReLU(inplace=True),)
            self.conv=nn.Sequential(
                                nn.Conv2d(middlechannel+inchannel_same,outchannel,3,padding=1),
                                nn.BatchNorm2d(outchannel),
                                nn.ReLU(inplace=True)
                                # nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
                              )

    def forward(self, uplayer,samlayer):
        self.up=self.block(uplayer)
        self.middle=torch.cat((self.up,samlayer),dim=1)
        self.out=self.conv(self.middle)
        return self.out
class crop_up(nn.Module):
    def __init__(self,inchannel_low,inchannel_same,middlechannel,outchannel,transpose=False):
        super(crop_up,self).__init__()
        if  transpose:
            self.block=nn.ConvTranspose2d(inchannel_low,middlechannel,3,2)
            self.conv=nn.Sequential(nn.Conv2d(inchannel_same+middlechannel,outchannel,3,padding=1),
                                nn.BatchNorm2d(outchannel),
                                nn.ReLU(inplace=True),
                                # nn.ConvTranspose2d(middlechannel,outchannel,3,2,1,1)
                                )
        else:
            self.block = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                       nn.Conv2d(inchannel_low, middlechannel, 3, padding=1),
                                       nn.BatchNorm2d(middlechannel),
                                       nn.ReLU(inplace=True), )
            self.conv=nn.Sequential(nn.Conv2d(inchannel_same+middlechannel,outchannel,3,padding=1),
                                nn.BatchNorm2d(outchannel),
                                nn.ReLU(inplace=True),

                                # nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
                              )
    def forward(self, uplayer,samlayer): #the uplayer need to be cropped and upsample
        tmp=self.block(uplayer)  # if block is transpose then need crop or it needs  pad(self.middle,(0,1,0,0),mode='replicate')
        uplayer=self.center_crop(tmp,samlayer)
        return self.conv(torch.cat((uplayer,samlayer),dim=1))
    def center_crop(self,img,target):
        h,w = img.shape[-2:]
        th, tw = target.shape[-2:]
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return img[...,i:i+th,j:j+tw]
class U_plus(nn.Module):
    def __init__(self):
        super(U_plus,self).__init__()
        self.convl0_0=vgg.features[:3]#64
        self.convl1_0=vgg.features[3:7] #128
        self.convl2_0=vgg.features[7:14] #256
        self.convl3_0=vgg.features[14:21] #512
        self.convl4_0=vgg.features[21:28] #512
        self.pool=vgg.features[-1]
        self.convl5_0=nn.Sequential(nn.Conv2d(512,512,3,padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True),
                                   # nn.Conv2d(middlechannel,middlechannel,3,padding=0),
                                   # nn.BatchNorm2d(middlechannel),
                                   # nn.ReLU(inplace=True),
                                  # nn.ConvTranspose2d(512, 256, 3, 2)
                                  )
        self.deconvl4_1=crop_up(512,512,256,512,True)
        self.deconvl3_1=up(512,512,256,512,True)
        self.deconvl2_1=up(512,256,256,256,True)
        self.deconvl1_1=up(256,128,128,128,True)
        self.deconvl0_1=up(128,64,64,64,True)
        self.deconvl3_2=up(512,512+512,256,512,True)
        self.deconvl2_2=up(512,256+256,256,256,True)
        self.deconvl2_3=up(512,256+256+256,256,256,True)
        self.deconvl1_2=up(256,128+128,128,128,True)
        self.deconvl1_3=up(256,128+128+128,128,128,True)
        self.deconvl1_4=up(256,128+128+128+128,128,128,True)
        self.deconvl0_2=up(128,64+64,64,64,True)
        self.deconvl0_3=up(128,64+64+64,64,64,True)
        self.deconvl0_4=up(128,64+64+64+64,64,64,True)
        self.deconvl0_5=up(128,64+64+64+64+64,64,64,True)
        self.f0=nn.Conv2d(64,1,1)
        self.f1=nn.Conv2d(64,1,1)
        self.f2=nn.Conv2d(64,1,1)
        self.f3=nn.Conv2d(64,1,1)
        self.f4=nn.Conv2d(64,1,1)


    def forward(self, input):
        self.l0_0=self.convl0_0(input)
        self.l1_0=self.convl1_0(self.l0_0)
        self.l2_0=self.convl2_0(self.l1_0)
        self.l3_0=self.convl3_0(self.l2_0)
        self.l4_0=self.convl4_0(self.l3_0)
        self.middle=self.convl5_0(self.pool(self.l4_0))

        self.l4_1=self.deconvl4_1(self.middle,self.l4_0)
        self.l3_1=self.deconvl3_1(self.l4_0,self.l3_0)
        self.l3_2=self.deconvl3_2(self.l4_1,torch.cat((self.l3_1,self.l3_0),dim=1))
        self.l2_1=self.deconvl2_1(self.l3_0,self.l2_0)
        self.l2_2=self.deconvl2_2(self.l3_1,torch.cat((self.l2_0,self.l2_1),dim=1))
        self.l2_3=self.deconvl2_3(self.l3_2,torch.cat((self.l2_0,self.l2_1,self.l2_2),dim=1))
        self.l1_1=self.deconvl1_1(self.l2_0,self.l1_0)
        self.l1_2=self.deconvl1_2(self.l2_1,torch.cat((self.l1_0,self.l1_1),dim=1))
        self.l1_3=self.deconvl1_3(self.l2_2,torch.cat((self.l1_0,self.l1_1,self.l1_2),dim=1))
        self.l1_4=self.deconvl1_4(self.l2_3,torch.cat((self.l1_0,self.l1_1,self.l1_2,self.l1_3),dim=1))
        self.l0_1=self.deconvl0_1(self.l1_0,self.l0_0)
        self.l0_2=self.deconvl0_2(self.l1_1,torch.cat((self.l0_0,self.l0_1),dim=1))
        self.l0_3=self.deconvl0_3(self.l1_2,torch.cat((self.l0_0,self.l0_1,self.l0_2),dim=1))
        self.l0_4=self.deconvl0_4(self.l1_3,torch.cat((self.l0_0,self.l0_1,self.l0_2,self.l0_3),dim=1))
        self.l0_5=self.deconvl0_5(self.l1_4,torch.cat((self.l0_0,self.l0_1,self.l0_2,self.l0_3,self.l0_4),dim=1))


        return self.f0(self.l0_1),self.f1(self.l0_2),self.f2(self.l0_3),self.f3(self.l0_4),self.f4(self.l0_5)


    def center_crop(self,img,target):
        h,w = img.shape[-2:]
        th, tw = target.shape[-2:]
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return img[...,i:i+th,j:j+tw]
# class UNET(nn.Module):
#     def __init__(self):
#         super(UNET,self).__init__()
#         self.conv1=vgg.features[:3] #64
#         self.conv2=vgg.features[3:7] #128
#         self.conv3=vgg.features[7:14] #256
#         self.conv4=vgg.features[14:21] #512
#         self.conv5=vgg.features[21:28] #512
#         self.pool=vgg.features[-1]
#         self.centre=nn.Sequential(nn.Conv2d(512,512,3,padding=1),
#                                    nn.BatchNorm2d(512),
#                                    nn.ReLU(inplace=True),
#                                    # nn.Conv2d(middlechannel,middlechannel,3,padding=0),
#                                    # nn.BatchNorm2d(middlechannel),
#                                    # nn.ReLU(inplace=True),
#                                   nn.ConvTranspose2d(512, 256, 3, 2)
#                                   )
#         # self.centre=deconv(512,512,256,True) #256
#         self.up5=deconv(768,384,192,True)
#         self.up4=deconv(704,352,176,True)
#         self.up3=deconv(432,216,108,True)
#         self.up2=deconv(236,118,60,True)
#         self.up1=nn.Conv2d(124,1,3,padding=1)
#     def forward(self, input):
#         self.layer1=self.conv1(input)
#         self.layer2=self.conv2(self.layer1)
#         self.layer3=self.conv3(self.layer2)
#         self.layer4=self.conv4(self.layer3)
#         self.layer5=self.conv5(self.layer4)
#
#         self.middle=self.centre(self.pool(self.layer5))
#         # print (self.layer1.shape)
#         # print (self.layer2.shape)
#         # print (self.layer3.shape)
#         # print (self.layer4.shape)
#         # print (self.layer5.shape)
#         # print ('midddle')
#         # print (self.middle.shape)
#         # self.middle=torch.nn.functional.pad(self.middle,(0,1,0,0),mode='replicate')
#         self.middle=self.center_crop(self.middle,self.layer5)
#         # print (self.middle.shape)
#         self.uplayer5 = self.up5 (torch.cat((self.middle,self.layer5),dim=1))
#         self.uplayer4 = self.up4 (torch.cat((self.uplayer5,self.layer4),dim=1))
#         self.uplayer3 = self.up3 (torch.cat((self.uplayer4,self.layer3),dim=1))
#         self.uplayer2 = self.up2 (torch.cat((self.uplayer3,self.layer2),dim=1))
#         self.uplayer1 = self.up1 (torch.cat((self.uplayer2,self.layer1),dim=1))
#         # print (self.uplayer5.shape)
#         # print (self.uplayer4.shape)
#         # print (self.uplayer3.shape)
#         # print (self.uplayer2.shape)
#         # print (self.uplayer1.shape)
#
#
#         return self.uplayer1
#     def center_crop(self,img,target):
#         h,w = img.shape[-2:]
#         th, tw = target.shape[-2:]
#         i = int(round((h - th) / 2.))
#         j = int(round((w - tw) / 2.))
#         return img[...,i:i+th,j:j+tw]
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
# def train(epoch):
#     model=UNET()
#     model.train()
#     model.to(device)
#     criterion=Diceloss()
#     optimize=torch.optim.Adam(model.parameters(),lr=0.0001)
#     for i in range(epoch):
#         tmp=0
#         for image,mask in trainload:
#             image,mask=image.to(device,dtype=dtype),mask.to(device,dtype=dtype)
#             optimize.zero_grad()
#             pred=model(image)
#             loss=criterion(pred,mask)
#             loss.backward()
#             optimize.step()
#             tmp=loss.data
#             # print ("loss ",tmp)
#             # break
#         print ("{0} epoch ,loss is {1}".format(i,tmp))
#     return model

def train(epoch):
    model=U_plus()
    model.train()
    model.to(device)
    criterion=Diceloss()
    optimize=torch.optim.Adam(model.parameters(),lr=0.0001)
    store_loss=[]
    for i in range(epoch):
        tmp=0
        for image,mask in trainload:
            image,mask=image.to(device,dtype=dtype),mask.to(device,dtype=dtype)
            optimize.zero_grad()
            l1,l2,l3,l4,l5=model(image)
            loss_list=list(map(lambda x,y:criterion(x,y),[l1,l2,l3,l4,l5],[mask]*5))
            tmp=reduce(lambda x,y:x+y,loss_list)
            loss=tmp/5
            loss.backward()
            optimize.step()
            tmp=loss.data
            print ("loss ",tmp)
            # break
        store_loss.append(tmp)
        print ("{0} epoch ,loss is {1}".format(i,tmp))
    return model,store_loss
def test(model):
    img=[]
    pred=[]
    mask=[]
    with torch.no_grad():
        model.eval()
        model.to(device)
        for image,mask_img in testload:
            image=image.to(device,dtype=dtype)
            output=model(image)
            label=output.cpu()>0.5
            pred.append(label.to(torch.long))
            img.append(image.cpu())
            mask.append(mask_img)
    return torch.cat(img),torch.cat(pred),torch.cat(mask)

def picture(img,pred,mask):
    # all must bu numpy object
    plt.figure()
    mean,std=np.array((0.485, 0.456, 0.406)),np.array((0.229, 0.224, 0.225))
    num=len(img)
    tmp=img.transpose(0,2,3,1)
    tmp=tmp*std+mean
    tmp=np.concatenate((tmp,pred,mask),axis=0)
    for i,j in enumerate(tmp,1):
        plt.subplot(3,num,i)
        plt.imshow(j)
    plt.show()
def torch_pic(img,pred,mask):
    img, pred, mask = img[:4], pred[:4].to(torch.long), mask[:4].to(torch.long)
    pred = pred.squeeze(dim=1)
    mask = mask.squeeze(dim=1)
    voc_colormap = [[0, 0, 0], [245, 222, 179]]
    voc_colormap = torch.from_numpy(np.array(voc_colormap))
    voc_colormap = voc_colormap.to(dtype)
    mean, std = np.array((0.485, 0.456, 0.406)), np.array((0.229, 0.224, 0.225))
    mean, std = torch.from_numpy(mean).to(dtype), torch.from_numpy(std).to(dtype)
    img = img.permute(0, 2, 3, 1)
    img = (img * std + mean)
    # pred=pred.permute(0,2,3,1)
    # mask=mask.permute(0,2,3,1)
    pred = voc_colormap[pred] / 255.0
    mask = voc_colormap[mask] / 255.0
    pred=pred.permute(0, 3, 1, 2)
    mask=mask.permute(0, 3, 1, 2)
    tmp = tv.utils.make_grid(torch.cat((img.permute(0,3,1,2), pred, mask)), nrow=4)
    plt.imshow(tmp.permute(1,2,0))
    plt.show()
def my_iou(label_pred,label_mask):
    iou=[]
    for i,j in zip(label_pred,label_mask):
        iou.append((i*j).sum()/(i.sum()+j.sum()-(i*j).sum()))
    return iou
model,loss_list=train(60)
torch.save(model.state_dict(),'uplus')
# model=U_plus()
# model.load_state_dict(torch.load('uplus'))
# img,pred,mask=test(model)
# ap,iou,hist=label_acc_score(mask,pred,2)
# iu=my_iou(pred,mask)
# torch_pic(img[10:14],pred[10:14].to(torch.long),mask[10:14].to(torch.long))

# a=torch.zeros(1,3,320,240)
# tmp=UNET()
# tmp(a)

#%%





# https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
# https://github.com/ybabakhin/kaggle_salt_bes_phalanx/blob/master/bes/losses.py
# https://arxiv.org/pdf/1606.04797.pdf#pdfjs.action=download
# okular
#https://arxiv.org/abs/1505.02496