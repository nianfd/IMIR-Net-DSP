import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
import scipy.misc
from PIL import Image
import pandas as pd
# import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import Dataset


import imageio
import cv2
import pdb

def random_unit(p):
    assert p >= 0 and p <= 1, "概率P的值应该处在[0,1]之间！"
    if p == 0:  # 概率为0，直接返回False
        return False
    if p == 1:  # 概率为1，直接返回True
        return True
    p_digits = len(str(p).split(".")[1])
    interval_begin = 1
    interval__end = pow(10, p_digits)
    R = random.randint(interval_begin, interval__end)
    if float(R)/interval__end < p:
        return True
    else:
        return False

# RGB-D
class Nutrition_RGBD(Dataset):
    def __init__(self, image_path, rgb_txt_dir, rgbd_txt_dir, training, transform=None):
        file_rgb = open(rgb_txt_dir, 'r')
        file_rgbd = open(rgbd_txt_dir, 'r')
        lines_rgb = file_rgb.readlines()
        lines_rgbd = file_rgbd.readlines()
        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        self.images_rgbd = []
        # pdb.set_trace()
        for line in lines_rgb:
            image_rgb = line.split()[0]  # side_angles/dish_1550862840/frames_sampled5/camera_A_frame_010.jpeg
            label = line.strip().split()[1]  # 类别 1-
            calories = line.strip().split()[2]
            mass = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image_rgb)]  # 每张图片路径
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        for line in lines_rgbd:
            image_rgbd = line.split()[0]
            self.images_rgbd += [os.path.join(image_path, image_rgbd)]

            # pdb.set_trace()
        # self.transform_rgb = transform[0]
        self.training = training
        self.transform = transform
        self.image_path = image_path

    # RGB-D  20210805
    def my_loader(path, Type):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if Type == 3:
                    img = img.convert('RGB')
                elif Type == 1:
                    img = img.convert('L')
                return img

    def __getitem__(self, index):
        img_rgb = Image.open(self.images[index])
        img_rgbd = Image.open(self.images_rgbd[index])
        ingredients_path = self.image_path + '/ingredients_fea_nonorm/' + self.images[index].split('/')[-2] + '.pth'
        #ingredients_path = self.image_path + '/ingredients_fea/' + self.images[index].split('/')[-2] + '.pth'
        ingredients_tensor = torch.load(ingredients_path)
        ingredients_tensor = torch.squeeze(ingredients_tensor)
        #print(ingredients_tensor.shape)
        # try:
        #     # img = cv2.resize(img, (self.imsize, self.imsize))
        #     img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))  # cv2转PIL
        #     img_rgbd = Image.fromarray(cv2.cvtColor(img_rgbd, cv2.COLOR_BGR2RGB))  # cv2转PIL
        # except:
        #     print("图片有误：", self.images[index])

        if self.training == True:
            if random_unit(0.5) == True:
                img_rgb = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
                img_rgbd = img_rgbd.transpose(Image.FLIP_LEFT_RIGHT)

            if random_unit(0.5) == True:
                rotate_degree = random.randint(0, 360)
                img_rgb = img_rgb.rotate(rotate_degree, expand = 1)
                img_rgbd = img_rgbd.rotate(rotate_degree, expand = 1)
                img_rgb = img_rgb.resize((416,416))
                img_rgbd = img_rgbd.resize((416, 416))


        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)



        return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], \
        self.total_carb[index], self.total_protein[index], img_rgbd, ingredients_tensor

    def __len__(self):
        return len(self.images)
