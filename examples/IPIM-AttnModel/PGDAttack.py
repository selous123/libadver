# Single Image Attack
#   for display only.

import libadver
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
from torchvision.transforms import transforms
from PIL import Image


modelFile = "/home/lrh/program/git/pytorch-example/adversarial-miccai2019/isic2016/IPMI2019-AttnMel/models/checkpoint.pth"
testCSVFile = "/home/lrh/program/git/pytorch-example/adversarial-miccai2019/isic2016/IPMI2019-AttnMel/test.csv"
print("======>load pretrained models")
net = AttnVGG(num_classes=2, attention=True, normalize_attn=True, vis = False)
# net = VGG(num_classes=2, gap=False)
checkpoint = torch.load(modelFile)
net.load_state_dict(checkpoint['state_dict'])
pretrained_clf = nn.DataParallel(net).cuda()
pretrained_clf.eval()

print("=======>load ISIC2016 dataset")
normalize = Normalize((0.7012, 0.5517, 0.4875), (0.0942, 0.1331, 0.1521))
transform_test = torch_transforms.Compose([
         RatioCenterCrop(0.8),
         Resize((256,256)),
         CenterCrop((224,224)),
         ToTensor(),
         normalize
    ])
testset = ISIC(csv_file=testCSVFile, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=4)


pgd_params = {
            'ord': np.inf,
            'y': None,
            'eps': 50.0 / 255,
            'eps_iter': 5.5 / 255,
            'nb_iter': 10,
            'rand_init': True,
            'rand_minmax': 50.0 / 255,
            'clip_min': 0.,
            'clip_max': 1.,
            'sanity_checks': True
        }

pgd_params['y'] = torch.LongTensor([1,1,1,1,1]).cuda()



PGDAttack = attack.ProjectGradientDescent(model = pretrained_clf)
isBenign = False
# benignRoot = "./adversarial_result/ori_img/benign"
# malignantRoot = "./adversarial_result/ori_img/malignant"
#
# benignImgs = [
#     "ISIC_0000234.jpg","ISIC_0000254.jpg", "ISIC_0000271.jpg",
#     "ISIC_0000325.jpg","ISIC_0000319.jpg"
# ]
# malignImgs = [
#     "ISIC_0000549.jpg", "ISIC_0001103.jpg", "ISIC_0001142.jpg",
#     "ISIC_0000547.jpg", "ISIC_0001100.jpg"
# ]
#
# if isBenign is True:
#     Imgs = benignImgs
#     Root = benignRoot
# else:
#     Imgs = malignImgs
#     Root = malignantRoot
#
#
mean = [0.7012, 0.5517, 0.4875]
std = [0.0942, 0.1331, 0.1521]
normalize = Normalize(mean, std)
transform_test = transforms.Compose([
         RatioCenterCrop(0.8),
         Resize((256,256)),
         CenterCrop((224,224)),
         ToTensor(),
         normalize
    ])
#
# images = torch.zeros([5,3,224,224])
# labels = torch.zeros([5])
#
# for batchIdx, Img in enumerate(Imgs):
#     benignPath = os.path.join(Root, Img)
#     img = Image.open(benignPath)
#     sample = {'image': img, 'image_seg': img, 'label': 0}
#     t_sample = transform_test(sample)
#     img = t_sample["image"]
#     #img.unsqueeze_(0)
#     images[batchIdx] = img
#     labels[batchIdx] = t_sample['label']
#     #print(images.shape)
# # print(img1.shape)
# images = images.cuda()
images, labels = load_data(isBenign, transform_test)
images = images.cuda()


pgd_params['clip_min'] = torch.min(images)
pgd_params['clip_max'] = torch.max(images)

# img1_temp = torch.zeros(img1.size())
# for c2 in range(3):
#     img1_temp.data[:,c2,:,:] = (img1.data[:,c2,:,:] * std[c2]) + mean[c2]
# torchvision.utils.save_image(img1_temp, "test/ori_img.jpg", normalize=True,  scale_each=True)
adv_images = PGDAttack.generate(images, **pgd_params)

#torchvision.utils.save_image(images, "adversarial_result/PGD/ori_img.png", normalize=True,  scale_each=True)
#torchvision.utils.save_image(adv_images, "adversarial_result/PGD/adv_img.png", normalize=True,  scale_each=True)
delta_ims = adv_images - images
print(torch.max(delta_ims) * 255)
#torchvision.utils.save_image(delta_ims, "adversarial_result/PGD/delta_im.png", normalize=True,  scale_each=True)


pred_test, a1, a2 = pretrained_clf(images)
print(torch.softmax(pred_test, dim=1))
predict = torch.argmax(pred_test, 1)
print(predict)

pred_test, a1, a2 = pretrained_clf(adv_images)
print(torch.softmax(pred_test, dim=1))
predict = torch.argmax(pred_test, 1)
print(predict)

images = images.cpu()
adv_images = adv_images.cpu()
delta_ims = delta_ims.cpu()

# for i in range(5):
image = images[4]
# image_PIL = libadver.visutils.recreate_image(image,mean,std)
# libadver.visutils.save_image(image_PIL,"adversarial_result/pre_img/malignant/ori_img_%d.png" %i)

adv_image = adv_images[4]
adv_image_PIL = libadver.visutils.recreate_image(adv_image,mean,std)
libadver.visutils.save_image(adv_image_PIL,"adversarial_result/PGD/malignant/adv_img_PGD.png")

delta_im = delta_ims[4].data.numpy()
libadver.visutils.save_gradient_images(delta_im,"adversarial_result/PGD/malignant/delta_im.png")
