import os
import copy
import random
import argparse
import time
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn

from SemanticSegmentation.loss import Loss
# from SemanticSegmentation.segnet import SegNet as segnet
from SemanticSegmentation.network import Segment
import sys
sys.path.append("..")

import cv2
from torchvision import transforms
from tools import transform

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))
        """
        if 'LFSD' in self.kwargs['datapath']:
            self.mean = np.array([[[128.67, 117.24, 107.97]]])
            self.std = np.array([[[66.14, 58.32, 56.37]]])
        elif 'NJUD' in self.kwargs['datapath']:
            self.mean = np.array([[[104.89, 101.66, 92.15]]])
            self.std = np.array([[[55.89, 53.03, 53.95]]])
        elif 'NLPR' in self.kwargs['datapath']:
            self.mean = np.array([[[126.74, 123.91, 123.04]]])
            self.std = np.array([[[52.91, 52.31, 50.61]]])
        elif 'STEREO797' in self.kwargs['datapath']:
            self.mean = np.array([[[113.17, 110.05, 98.60]]])
            self.std = np.array([[[58.60, 55.89, 58.32]]])
        """
        # else:
        # raise ValueError

        """
        self.mean = np.array([[[0.485, 0.456, 0.406]]])*255.0
        self.std = np.array([[[0.229, 0.224, 0.225]]])*255.0

        """
        self.mean = np.array([[[128.67, 117.24, 107.97]]])
        self.std = np.array([[[66.14, 58.32, 56.37]]])
        self.d_mean = 116.09
        self.d_std = 56.61

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None

def execSS(rgb, depth):
    mean = np.array([[[128.67, 117.24, 107.97]]])
    std = np.array([[[66.14, 58.32, 56.37]]])
    d_mean = 116.09
    d_std = 56.61
    # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_transform = transform.Compose(
        transform.Resize((256, 256)),
        transform.Normalize(mean=mean, std=std, d_mean=d_mean, d_std=d_std),
        transform.ToTensor(depth_gray=True))

    # model = segnet()
    cfg = Config(datapath="/project/1_2301/DPANet-master/YCB_Video_Dataset/data/0000",
                 snapshot="/project/1_2301/DPANet-master/new_model.pth", mode='test')
    # 参数 snapshot="res50_res/model-1"  指定的训练好的模型
    model = Segment(backbone='resnet50', cfg=cfg, norm_layer=nn.BatchNorm2d) # 模型里面会加载指定的训练好的模型
    model = model.cuda()
    model.eval()

    rgb, depth = eval_transform(rgb, depth)
    rgb = rgb.cuda()
    depth = depth.cuda()

    # #print('rgb', type(rgb))
    # #print('rgb', rgb.size)
    # rgb = np.asarray(rgb)
    # #print('rgb', rgb.shape)
    # rgb = np.transpose(rgb, (2, 0, 1))
    # #print('rgb', rgb.shape)
    # #print('rgb', type(rgb))
    # rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
    # #print('rgb', rgb.shape)
    # #print('rgb', type(rgb))
    # rgb = rgb.unsqueeze(0)
    # #print('rgb', rgb.shape)
    # #print('rgb', type(rgb))

    # rgb = Variable(rgb).cuda()  # tensor 1 3 480 640
    rgb = rgb.unsqueeze(0)
    depth = depth.unsqueeze(0)
    semantic, _ = model(rgb, depth)  # 1 22 480 640
    new_out = semantic.detach().cpu().numpy()
    new_out = new_out[0][0]
    new_out = (new_out - new_out.min()) / (new_out.max() - new_out.min())
    new_out = np.resize(new_out, (480, 640))
    new_out = new_out * 21  # 标签只有21类
    new_out = new_out.astype(np.uint8)
    #print('semantic', semantic.shape)

    # convert output tensor to masked image
    # seg_data = semantic[0]  # 22 480 640
    # seg_data2 = torch.transpose(seg_data, 0, 2) # 640 480 22
    # seg_data2 = torch.transpose(seg_data2, 0, 1)  # 480 640 22
    # seg_image = torch.argmax(seg_data2, dim=-1)  # 480 640  torch.int64
    # obj_list = torch.unique(seg_image).detach().cpu().numpy()
    # seg_image = seg_image.detach().cpu().numpy()
    #print('seg_image', seg_image.shape)
    seg_image = new_out
    image = seg_image.astype('uint8')
    #print('image', image.shape)
    medianblur = cv2.medianBlur(image, ksize=3)
    #print('medianblur', medianblur.shape)
    #dillate = cv2.dilate(medianblur, kernel=(5, 5))
    dillate = cv2.dilate(medianblur, kernel=(7, 7))
    #print('dillate', dillate.shape)
    dillate_image = Image.fromarray(dillate.astype('uint8'))
    #print('dillate_image', dillate_image.size)

    return dillate_image

# def execSS(rgb):
#
#     norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
#     model = segnet()
#     model.load_state_dict(torch.load("/project/1_2301/MaskedFusion-master/SemanticSegmentation/trained_models/model_9_0.556849330663681.pth"))
#     model = model.cuda()
#     model.eval()
#
#     #print('rgb', type(rgb))
#     #print('rgb', rgb.size)
#     rgb = np.asarray(rgb)
#     #print('rgb', rgb.shape)
#     rgb = np.transpose(rgb, (2, 0, 1))
#     #print('rgb', rgb.shape)
#     #print('rgb', type(rgb))
#     rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
#     #print('rgb', rgb.shape)
#     #print('rgb', type(rgb))
#     rgb = rgb.unsqueeze(0)
#     #print('rgb', rgb.shape)
#     #print('rgb', type(rgb))
#
#     rgb = Variable(rgb).cuda()  # tensor 1 3 480 640
#     semantic = model(rgb)  # 1 22 480 640
#     #print('semantic', semantic.shape)
#
#     # convert output tensor to masked image
#     seg_data = semantic[0]  # 22 480 640
#     seg_data2 = torch.transpose(seg_data, 0, 2) # 640 480 22
#     seg_data2 = torch.transpose(seg_data2, 0, 1)  # 480 640 22
#     seg_image = torch.argmax(seg_data2, dim=-1)  # 480 640  torch.int64
#     obj_list = torch.unique(seg_image).detach().cpu().numpy()
#     seg_image = seg_image.detach().cpu().numpy()
#     #print('seg_image', seg_image.shape)
#
#     image = seg_image.astype('uint8')
#     #print('image', image.shape)
#     medianblur = cv2.medianBlur(image, ksize=3)
#     #print('medianblur', medianblur.shape)
#     #dillate = cv2.dilate(medianblur, kernel=(5, 5))
#     dillate = cv2.dilate(medianblur, kernel=(7, 7))
#     #print('dillate', dillate.shape)
#     dillate_image = Image.fromarray(dillate.astype('uint8'))
#     #print('dillate_image', dillate_image.size)
#
#     return dillate_image
