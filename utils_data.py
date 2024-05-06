import logging
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.transforms.transforms import CenterCrop
from mydataset import Nutrition_RGBD
import pdb
import random

def get_DataLoader(args):
    #image_sizes = ((256, 352), (320, 448))
    train_transform = transforms.Compose([
        #transforms.Resize((320, 448)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    test_transform = transforms.Compose([
        #transforms.Resize((320, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    nutrition_rgbd_ims_root = os.path.join(args.data_root, 'imagery')
    nutrition_train_txt = os.path.join(args.data_root, 'imagery','rgbd_train_processed_refine.txt')
    nutrition_test_txt = os.path.join(args.data_root, 'imagery','rgbd_test_processed1_refine.txt') # depth_color.png
    nutrition_train_rgbd_txt = os.path.join(args.data_root, 'imagery','rgb_in_overhead_train_processed_refine.txt')
    nutrition_test_rgbd_txt = os.path.join(args.data_root, 'imagery','rgb_in_overhead_test_processed1_refine.txt') # rbg.png
    trainset = Nutrition_RGBD(nutrition_rgbd_ims_root, nutrition_train_rgbd_txt, nutrition_train_txt, training = True, transform = train_transform)
    testset = Nutrition_RGBD(nutrition_rgbd_ims_root, nutrition_test_rgbd_txt, nutrition_test_txt, training = False, transform = test_transform)

    train_loader = DataLoader(trainset,
                              batch_size=args.b,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True
                              )
    test_loader = DataLoader(testset,
                             batch_size=args.b,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True
                             )

    return train_loader, test_loader



