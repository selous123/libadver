import libadver.attack as attack
import libadver.models.generators as generators
import torch.optim as optim
import os
import csv
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as utils
import torchvision.transforms as torch_transforms
from networks import AttnVGG, VGG
from loss import FocalLoss
from data import preprocess_data_2016, preprocess_data_2017, ISIC
from utilities import *
from transforms import *
import libadver.attack as attack

print("======>load pretrained models")
net = AttnVGG(num_classes=2, attention=True, normalize_attn=True, vis = False)
# net = VGG(num_classes=2, gap=False)
checkpoint = torch.load('models/checkpoint.pth')
net.load_state_dict(checkpoint['state_dict'])
pretrained_clf = nn.DataParallel(net).cuda()
pretrained_clf.eval()

print("=======>load ISIC2016 dataset")
mean = (0.7012, 0.5517, 0.4875)
std = (0.0942, 0.1331, 0.1521)
normalize = Normalize(mean, std)
transform = torch_transforms.Compose([
         RatioCenterCrop(0.8),
         Resize((256,256)),
         CenterCrop((224,224)),
         ToTensor(),
         normalize
    ])
testset = ISIC(csv_file='test.csv', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=4)

trainset = ISIC(csv_file='train.csv', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8, drop_last=True)

print(len(trainloader))

isTrain = False

params = {
        "attackModelPath" : None,
        "mag_in" : 7.0,
        "ord" : "inf",
        "epochNum" : 3,
        "criterion" : nn.CrossEntropyLoss(),
        "ncInput" : 3,
        "ncOutput" : 3,
        "mean" : mean,
        "std" : std,
        "MaxIter" : 100
    }
print(params)
saveModelPath = "adversarial_result/GAP_im_m7n3.pth"
attackModel = generators.define(input_nc = params["ncInput"], output_nc = params["ncOutput"],
                                ngf = 64, gen_type = "unet", norm="batch", act="relu", gpu_ids = [0])


if isTrain is True:
    print("===>Train")
    optimizerG = optim.Adam(attackModel.parameters(), lr = 2e-4, betas = (0.5, 0.999))
    params["optimizerG"] = optimizerG
    GAPAttack = attack.GenerativeAdversarialPerturbations(pretrained_clf, attackModel, **params)
    GAPAttack.train(trainloader, saveModelPath)
else:
    print("===>Test")
    ## test
    params["attackModelPath"] = saveModelPath
    GAPAttack = attack.GenerativeAdversarialPerturbations(pretrained_clf, attackModel, **params)
    correct = 0
    total = 0
    for i, data in enumerate(testloader):
        images, labels = data['image'], data['label']
        images, labels = images.cuda(), labels.cuda()
        adv_images = GAPAttack.generate(images)
        predicted,_,_ = pretrained_clf(adv_images)
        predicted_labels = torch.argmax(predicted,1)
        #print(predicted_labels)
        correct += torch.sum(predicted_labels.eq(labels))
        #print(targets)
        total += images.shape[0]
        print("ACC:%.3f | %d,%d" %(100.0*float(correct) / total, correct, total))
