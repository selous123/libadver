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
from data import preprocess_data_2016, preprocess_data_2017, ISIC, load_data
from utilities import *
from transforms import *
import libadver.attack as attack
import libadver

print("======>load pretrained models")
net = AttnVGG(num_classes=2, attention=True, normalize_attn=True, vis = False)
# net = VGG(num_classes=2, gap=False)
modelFile = "/home/lrh/program/git/pytorch-example/adversarial-miccai2019/isic2016/IPMI2019-AttnMel/models/checkpoint.pth"
testCSVFile = "/home/lrh/program/git/pytorch-example/adversarial-miccai2019/isic2016/IPMI2019-AttnMel/test.csv"
trainCSVFile = "/home/lrh/program/git/pytorch-example/adversarial-miccai2019/isic2016/IPMI2019-AttnMel/train.csv"

checkpoint = torch.load(modelFile)
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
testset = ISIC(csv_file=testCSVFile, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=4)

trainset = ISIC(csv_file=trainCSVFile, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8, drop_last=True)

print(len(trainloader))

isTrain = False
isTestDataset = True

params = {
        "attackModelPath" : None,
        "mag_in" : 13.0,
        "ord" : "inf",
        "epochNum" : 2,
        "criterion" : nn.CrossEntropyLoss(),
        "ncInput" : 3,
        "ncOutput" : 3,
        "mean" : mean,
        "std" : std,
        "MaxIter" : 100
    }
print(params)
saveModelPath = "adversarial_result/GAP/GAP_im_m13n2_5.pth"
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

    if isTestDataset:
        # with open('test_results.csv', 'wt', newline='') as csv_file:
        #     csv_writer = csv.writer(csv_file, delimiter=',')
        #     for i, data in enumerate(testloader, 0):
        #         images_test, labels_test = data['image'], data['label']
        #         images_test, labels_test = images_test.cuda(), labels_test.cuda()
        #         pred_test, __, __ = pretrained_clf.forward(images_test)
        #         predict = torch.argmax(pred_test, 1)
        #         total += labels_test.size(0)
        #         correct += torch.eq(predict, labels_test).sum().double().item()
        #         # record test predicted responses
        #         responses = F.softmax(pred_test, dim=1).squeeze().detach().cpu().numpy()
        #         responses = [responses[i] for i in range(responses.shape[0])]
        #         csv_writer.writerows(responses)
        for i, data in enumerate(testloader):
            images, labels = data['image'], data['label']
            images, labels = images.cuda(), labels.cuda()
            adv_images = GAPAttack.generate(images)
            #adv_images = images
            predicted,_,_ = pretrained_clf(adv_images)
            predicted_labels = torch.argmax(predicted,1)
            #print(predicted_labels)
            correct += torch.sum(predicted_labels.eq(labels))
            #print(targets)
            total += images.shape[0]
            print("ACC:%.3f | %d,%d" %(100.0*float(correct) / total, correct, total))
            for c2 in range(3):
                adv_images.data[:,c2,:,:] = (adv_images.data[:,c2,:,:] * std[c2]) + mean[c2]
                images.data[:,c2,:,:] = (images.data[:,c2,:,:] * std[c2]) + mean[c2]
            post_l_inf = (adv_images.data - images[0:adv_images.size(0)].data).abs().max() * 255.0
            #print("Specified l_inf: ", params["mag_in"], ", maximum l_inf of generated perturbations: ", post_l_inf)

    else:
        images, labels = load_data(isBenign = True, transform = transform)
        adv_images = GAPAttack.generate(images)
        predicted,_,_ = pretrained_clf(images)
        print(torch.softmax(predicted,1))
        predicted,_,_ = pretrained_clf(adv_images)
        print(torch.softmax(predicted,1))
        predicted_labels = torch.argmax(predicted,1)
        #print(predicted_labels)
        correct += torch.sum(predicted_labels.eq(labels))
        #print(targets)
        total += images.shape[0]
        print("ACC:%.3f | %d,%d" %(100.0*float(correct) / total, correct, total))


        ###save image
        delta_ims = adv_images - images
        images = images.cpu()
        adv_images = adv_images.cpu()
        delta_ims = delta_ims.cpu()

        for c2 in range(3):
            adv_images.data[:,c2,:,:] = (adv_images.data[:,c2,:,:] * std[c2]) + mean[c2]
            images.data[:,c2,:,:] = (images.data[:,c2,:,:] * std[c2]) + mean[c2]
            delta_ims.data[:,c2,:,:] = (delta_ims.data[:,c2,:,:] * std[c2]) + mean[c2]


        #post_l_inf = (adv_images.data - images[0:adv_images.size(0)].data).abs().max() * 255.0
        #print("Specified l_inf: ", params["mag_in"], ", maximum l_inf of generated perturbations: ", post_l_inf)
        for c2 in range(3):
            delta_ims.data[:,c2,:,:] = delta_ims.data[:,c2,:,:]  - mean[c2]
        print(torch.max(delta_ims) * 255)
        ## Black
        #delta_ims = adv_images - images
        torchvision.utils.save_image(delta_ims.data, 'delta_im.png')
        # adv_image = adv_images[4]
        # adv_image_PIL = libadver.visutils.recreate_image(adv_image,mean,std)
        # libadver.visutils.save_image(adv_image_PIL,"adversarial_result/GAP/malignant/adv_img_4.png")
        #
        # #delta_im = delta_ims[0].data.numpy()
        # delta_im = delta_ims[4]
        # #libadver.visutils.save_gradient_images(delta_im,"adversarial_result/GAP/benign/delta_im_0.png")
        # delta_im_PIL = libadver.visutils.recreate_image(delta_im,mean,std)
        # libadver.visutils.save_image(delta_im_PIL,"adversarial_result/GAP/malignant/delta_im_4.png")
