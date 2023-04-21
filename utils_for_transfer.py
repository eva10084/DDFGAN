import torch
from torch import nn
from torch.utils.data import Dataset
import os
import SimpleITK as sitk
#import nibabel as nib
import numpy as np
import glob
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.backends import cudnn
from torch import optim
import torchvision
import torchvision.transforms as transforms
import time
import random
from skimage import transform
import torch.nn.init as init


# 目前只能是源域C0，目标域LGE

source = 'C0'
target = 'LGE'


if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU 可用
else:
    device = torch.device("cpu")   # 只能使用 CPU




class target_TrainSet(Dataset):
    def __init__(self,extra):
        self.imgdir = extra+'/' + target + '/'

        self.imgsname = glob.glob(self.imgdir + '*' + target + '.nii' + '*')

        imgs = np.zeros((1,192,192))
        self.info = []
        for img_num in range(len(self.imgsname)):
            itkimg = sitk.ReadImage(self.imgsname[img_num].replace('\\', '/'))
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
            npimg = npimg.astype(np.float32)

            imgs = np.concatenate((imgs,npimg),axis=0)
            spacing = itkimg.GetSpacing()[2]
            media_slice = int(npimg.shape[0] / 2)
            for i in range(npimg.shape[0]):
                a, _ = divmod((i - media_slice) * spacing, 20.0)
                info = int(a) + 3
                if info < 0:
                    info = 0
                elif info > 5:
                    info = 5

                self.info.append(info)
        self.imgs = imgs[1:,:,:]
        # print (imgs.shape)

    def __getitem__(self, item):
        imgindex,crop_indice = divmod(item,4)

        npimg = self.imgs[imgindex,:,:]
        randx = np.random.randint(-16,16)
        randy = np.random.randint(-16, 16)
        npimg=npimg[96+randx-80:96+randx+80,96+randy-80:96+randy+80]

        # npimg_o = transform.resize(npimg, (80, 80),
        #                      order=3, mode='edge', preserve_range=True)
        #npimg_resize = transform.resize(npimg, (96, 96), order=3,mode='edge', preserve_range=True)
        npimg_down2 = transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(npimg, (40,40 ), order=3,mode='edge', preserve_range=True)

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor),torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)

    def __len__(self):

        return self.imgs.shape[0]*4


class source_TrainSet(Dataset):
    def __init__(self,extra):
        self.imgdir = extra+'/' + source +'/'  # 加文件夹名

        # 获取一个路径列表，这些路径是指定目录下所有以 C0.nii 结尾的文件
        self.imgsname = glob.glob(self.imgdir + '*' + source + '.nii' + '*')  # 图片

        imgs = np.zeros((1,192,192))
        labs = np.zeros((1,192,192))
        self.info = []
        for img_num in range(len(self.imgsname)):
            itkimg = sitk.ReadImage(self.imgsname[img_num].replace('\\', '/'))
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1

            imgs = np.concatenate((imgs,npimg),axis=0)

            labname = self.imgsname[img_num].replace('.nii','_manual.nii')   # 获得对应图片的标注名
            itklab = sitk.ReadImage(labname)
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

            labs = np.concatenate((labs, nplab), axis=0)

            spacing = itkimg.GetSpacing()[2]
            media_slice = int(npimg.shape[0] / 2)
            for i in range(npimg.shape[0]):
                a, _ = divmod((i - media_slice) * spacing, 20.0)
                info = int(a) + 3
                if info < 0:
                    info = 0
                elif info > 5:
                    info = 5

                self.info.append(info)
        self.imgs = imgs[1:,:,:]
        self.labs = labs[1:,:,:]
        self.imgs.astype(np.float32)
        self.labs.astype(np.float32)



    def __getitem__(self, item):
        imgindex,crop_indice = divmod(item,4)

        npimg = self.imgs[imgindex,:,:]
        nplab = self.labs[imgindex,:,:]

        # npimg = transform.resize(npimg, (96, 96), order=3,mode='edge', preserve_range=True)
        # nplab = transform.resize(nplab, (96, 96), order=0,mode='edge', preserve_range=True)
        randx = np.random.randint(-16,16)
        randy = np.random.randint(-16, 16)
        npimg=npimg[96+randx-80:96+randx+80,96+randy-80:96+randy+80]
        nplab=nplab[96+randx-80:96+randx+80,96+randy-80:96+randy+80]

        # npimg_o=transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        # nplab_o=transform.resize(nplab, (80,80 ), order=0,mode='edge', preserve_range=True)

        npimg_down2 = transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(npimg, (40,40 ), order=3,mode='edge', preserve_range=True)

        nplab_down2 = transform.resize(nplab, (80,80 ), order=0,mode='edge', preserve_range=True)
        nplab_down4 = transform.resize(nplab, (40,40), order=0,mode='edge', preserve_range=True)

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(nplab).type(dtype=torch.LongTensor),torch.from_numpy(nplab_down2).type(dtype=torch.LongTensor),torch.from_numpy(nplab_down4).type(dtype=torch.LongTensor),torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)

    def __len__(self):

        return self.imgs.shape[0]*4

