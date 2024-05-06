from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import numpy as np
from utils import *
from utils_data import get_DataLoader
#from light_cnn_v4 import LightCNN_V4
import torchvision as tv
from mynetwork import MyResNetRGBD
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1, conv3x3
import logging
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.transforms.transforms import CenterCrop
from mydataset import Nutrition_RGBD
import pdb
import random
import csv
test_transform = transforms.Compose([
    # transforms.Resize((320, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_root= '/home/user/nfdProject/food/'
nutrition_rgbd_ims_root = os.path.join(data_root, 'imagery')
nutrition_test_txt = os.path.join(data_root, 'imagery', 'rgbd_test_processed1_refine.txt')  # depth_color.png
nutrition_test_rgbd_txt = os.path.join(data_root, 'imagery','rgb_in_overhead_test_processed1_refine.txt')  # rbg.png
testset = Nutrition_RGBD(nutrition_rgbd_ims_root, nutrition_test_rgbd_txt, nutrition_test_txt, training=False,
                         transform=test_transform)

testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model = MyResNetRGBD()
model = model.to(device)

checkpoint_path = './saved/ckpt_best.pth'
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

model.eval()

epoch_iterator = tqdm(testloader,
                      desc="Testing... (loss=X.X)",
                      bar_format="{l_bar}{r_bar}",
                      dynamic_ncols=True)

csv_rows = []

with torch.no_grad():
    for batch_idx, x in enumerate(epoch_iterator):
        inputs = x[0].to(device)
        total_calories = x[2].to(device).float()
        total_mass = x[3].to(device).float()
        total_fat = x[4].to(device).float()
        total_carb = x[5].to(device).float()
        total_protein = x[6].to(device).float()
        inputs_rgbd = x[7].to(device)
        inputs_ingredients = x[8].to(device)


        outputs, rgb_fea_t = model(inputs, inputs_rgbd)
        #print(x[1])
        for i in range(len(x[1])):  # IndexError: tuple index out of range  最后一轮的图片数量不到32，不能被batchsiz
            dish_id = x[1]
            calories = outputs[0]
            mass = outputs[1]
            fat = outputs[2]
            carb = outputs[3]
            protein = outputs[4]
            dish_row = [dish_id, calories.item(), mass.item(), fat.item(), carb.item(), protein.item()]
            csv_rows.append(dish_row)

new_csv_rows = []
predict_values = dict()
# pdb.set_trace()
key = ''
for iterator in csv_rows:
    if key != iterator[0][0]:
        key = iterator[0][0]
        #print(key)
        predict_values[key] = []
        predict_values[key].append(iterator[1:])
    else:
        predict_values[key].append(iterator[1:])
# pdb.set_trace()
for k,v in predict_values.items():
    nparray = np.array(v)
    predict_values[k] = np.mean(nparray,axis=0) #每列求均值
    new_csv_rows.append([k, predict_values[k][0], predict_values[k][1], predict_values[k][2], predict_values[k][3], predict_values[k][4]])

headers = ["dish_id", "calories", "mass", "fat", "carb", "protein"]
csv_file_path = "epoch_result_dish.csv"
#每张图片的结果写入csv
#每个dish写入csv
with open(csv_file_path,'w')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(new_csv_rows)
