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

parser = argparse.ArgumentParser(description='PyTorch Light CNN Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', type=int, default=200,help="the frequency of write to logtxt" )
parser.add_argument('--model', default='rgbonly', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29, LightCNN-29v2')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--data_root', default='C:/project/data/nutrition5k', type=str, metavar='PATH',
#                     help='path to root path of images (default: none)')
parser.add_argument('--data_root', default='/home/user/nfdProject/food', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument('--train_list', default='', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--val_list', default='', type=str, metavar='PATH',
                    help='path to validation list (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--run_name',type=str, default="resnet50roi2level")

args = parser.parse_args()
log_file_path = os.path.join("./logs",f'checkpoint_{args.model}_{args.run_name}',"train_log.txt")
check_dirs(os.path.join("./logs",f'checkpoint_{args.model}_{args.run_name}'))
logtxt(log_file_path, str(vars(args)))



def main():
    #global args
    #model = LightCNN_V4()
    # model = MyResNet(Bottleneck, [3, 4, 6, 3])  # 这里具体的参数参考库中源代码
    # model.load_state_dict(torch.load('resnet50-19c8e357.pth'), strict=False)
    model = MyResNetRGBD()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    print(model)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #load image
    train_loader, test_loader = get_DataLoader(args)

    # define loss function and optimizer
    criterion = nn.L1Loss()

    criterion = criterion.to(device)

    #criterion_2 = nn.MSELoss(size_average = False)
    #criterion_2 = nn.MSELoss()
    criterion_2 = nn.L1Loss()
    criterion_2 = criterion_2.to(device)


    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        validate(test_loader, model, criterion, device, epoch)
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, criterion_2, optimizer, device, epoch)


        #scheduler.step()
        # evaluate on validation set
        #prec1 = validate(test_loader, model, criterion, device)

def train(train_loader, model, criterion, criterion_2, optimizer, device, epoch):
    model.train()
    train_loss = 0
    calories_loss = 0
    mass_loss = 0
    fat_loss = 0
    carb_loss = 0
    protein_loss = 0
    ingredients_loss = 0

    epoch_iterator = tqdm(train_loader,
                          desc="Training (X / X Steps) (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    for batch_idx, x in enumerate(epoch_iterator):  # (inputs, targets,ingredient)
        '''Portion Independent Model'''
        inputs = x[0].to(device)
        total_calories = x[2].to(device).float()
        total_mass = x[3].to(device).float()
        total_fat = x[4].to(device).float()
        total_carb = x[5].to(device).float()
        total_protein = x[6].to(device).float()
        inputs_rgbd = x[7].to(device)
        inputs_ingredients = x[8].to(device)

        optimizer.zero_grad()

        outputs, rgb_fea_t = model(inputs, inputs_rgbd)
        total_calories_loss = total_calories.shape[0] * criterion(outputs[0], total_calories) / total_calories.sum().item()
        total_mass_loss = total_calories.shape[0] * criterion(outputs[1], total_mass) / total_mass.sum().item()
        total_fat_loss = total_calories.shape[0] * criterion(outputs[2], total_fat) / total_fat.sum().item()
        total_carb_loss = total_calories.shape[0] * criterion(outputs[3], total_carb) / total_carb.sum().item()
        total_protein_loss = total_calories.shape[0] * criterion(outputs[4], total_protein) / total_protein.sum().item()
        total_ingredients_loss = criterion_2(rgb_fea_t, inputs_ingredients)
        #loss = total_calories_loss + total_mass_loss + total_fat_loss + total_carb_loss + total_protein_loss + 0.001 * total_ingredients_loss
        loss = total_calories_loss + total_mass_loss + total_fat_loss + total_carb_loss + total_protein_loss + 0.001 * total_ingredients_loss

        #loss = total_calories_loss + total_mass_loss + total_fat_loss + total_carb_loss + total_protein_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        calories_loss += total_calories_loss.item()
        mass_loss += total_mass_loss.item()
        fat_loss += total_fat_loss.item()
        carb_loss += total_carb_loss.item()
        protein_loss += total_protein_loss.item()
        ingredients_loss += total_ingredients_loss.item()

        if (batch_idx+1) % args.print_freq == 0 or batch_idx+1 == len(train_loader):
            logtxt(log_file_path, 'Epoch: [{}][{}/{}]\t'
                    'Loss: {:2.5f} \t'
                    'calorieloss: {:2.5f} \t'
                    'massloss: {:2.5f} \t'
                    'fatloss: {:2.5f} \t'
                    'carbloss: {:2.5f} \t'
                    'proteinloss: {:2.5f} \t'
                    'ingredientsloss: {:2.5f} \n'.format(
                    epoch, batch_idx+1, len(train_loader),
                    train_loss/(batch_idx+1),
                    calories_loss/(batch_idx+1),
                    mass_loss/(batch_idx+1),
                    fat_loss/(batch_idx+1),
                    carb_loss/(batch_idx+1),
                    protein_loss/(batch_idx+1),
                    ingredients_loss/(batch_idx+1)))

best_loss = 10000
def validate(test_loader, model, criterion, device, epoch):
    # switch to evaluate mode
    global best_loss
    model.eval()
    test_loss = 0
    calories_loss = 0
    mass_loss = 0
    fat_loss = 0
    carb_loss = 0
    protein_loss = 0

    epoch_iterator = tqdm(test_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)

    with torch.no_grad():
        for batch_idx, x in enumerate(epoch_iterator):  # testloader
            inputs = x[0].to(device)
            total_calories = x[2].to(device).float()
            total_mass = x[3].to(device).float()
            total_fat = x[4].to(device).float()
            total_carb = x[5].to(device).float()
            total_protein = x[6].to(device).float()
            inputs_rgbd = x[7].to(device)

            outputs,_ = model(inputs, inputs_rgbd)

            # loss
            calories_total_loss = total_calories.shape[0] * criterion(outputs[0], total_calories) / total_calories.sum().item()
            mass_total_loss = total_calories.shape[0] * criterion(outputs[1], total_mass) / total_mass.sum().item()
            fat_total_loss = total_calories.shape[0] * criterion(outputs[2], total_fat) / total_fat.sum().item()
            carb_total_loss = total_calories.shape[0] * criterion(outputs[3], total_carb) / total_carb.sum().item()
            protein_total_loss = total_calories.shape[0] * criterion(outputs[4], total_protein) / total_protein.sum().item()

            loss = calories_total_loss + mass_total_loss + fat_total_loss + carb_total_loss + protein_total_loss

            test_loss += loss.item()
            calories_loss += calories_total_loss.item()
            mass_loss += mass_total_loss.item()
            fat_loss += fat_total_loss.item()
            carb_loss += carb_total_loss.item()
            protein_loss += protein_total_loss.item()

        epoch_iterator.set_description(
            "Testing Epoch[%d] | loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f" % (
            epoch, test_loss / (batch_idx + 1), calories_loss / (batch_idx + 1), mass_loss / (batch_idx + 1),
            fat_loss / (batch_idx + 1), carb_loss / (batch_idx + 1), protein_loss / (batch_idx + 1)))

        logtxt(log_file_path, 'Test Epoch: [{}][{}/{}]\t'
                              'Loss: {:2.5f} \t'
                              'calorieloss: {:2.5f} \t'
                              'massloss: {:2.5f} \t'
                              'fatloss: {:2.5f} \t'
                              'carbloss: {:2.5f} \t'
                              'proteinloss: {:2.5f} \n'.format(
            epoch, batch_idx + 1, len(test_loader),
                   test_loss / len(test_loader),
                   calories_loss / len(test_loader),
                   mass_loss / len(test_loader),
                   fat_loss / len(test_loader),
                   carb_loss / len(test_loader),
                   protein_loss / len(test_loader)))
        # Save checkpoint.
        # pdb.set_trace()
    if best_loss > test_loss:
        best_loss = test_loss
        print('Saving..')
        state = {
            'state_dict': model.state_dict(),
            'epoch': epoch
        }
        savepath = f"./saved/regression_{args.model}_{args.run_name}"
        check_dirs(savepath)
        torch.save(state, os.path.join(savepath, f"ckpt_best.pth"))

def adjust_learning_rate(optimizer, epoch):
    scale = 0.5
    step  = 45
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale

if __name__ == '__main__':
    main()
