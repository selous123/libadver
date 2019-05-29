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
import libadver.attack as attack
from torchvision.transforms import transforms
from PIL import Image
def data_reader(transform=None):
    visRoot = "./adversarial_result/vis"
    imgs = [
        "adv_img_GAP.png","adv_img_PGD.png", "ori_img.jpg"
    ]
    images = torch.zeros([3,3,224,224])
    for batchIdx, img in enumerate(imgs):
        imgPath = os.path.join(visRoot, img)
        img = Image.open(imgPath)
        sample = {'image': img, 'image_seg': img, 'label': 1}

        if transform is not None:
            t_sample = transform(sample)
        #img.unsqueeze_(0)
        img = t_sample["image"]
        images[batchIdx] = img
    return images

normalize = Normalize((0.7012, 0.5517, 0.4875), (0.0942, 0.1331, 0.1521))
transform_test = torch_transforms.Compose([
         ToTensor(),
         normalize
    ])

images = data_reader(transform_test)
images = images.cuda()


modelFile = "/home/lrh/program/git/pytorch-example/adversarial-miccai2019/isic2016/IPMI2019-AttnMel/models/checkpoint.pth"
testCSVFile = "/home/lrh/program/git/pytorch-example/adversarial-miccai2019/isic2016/IPMI2019-AttnMel/test.csv"
print("======>load pretrained models")
# net = AttnVGG(num_classes=2, attention=True, normalize_attn=True, vis = False)
# # net = VGG(num_classes=2, gap=False)
# checkpoint = torch.load(modelFile)
# net.load_state_dict(checkpoint['state_dict'])
# pretrained_clf = nn.DataParallel(net).cuda()
# pretrained_clf.eval()

# attention map
# fileNames = ["GAP","PGD","ori"]
# for i in range(3):
#     image = images[i].unsqueeze(0)
#     predict, a1, a2 = pretrained_clf(image)
#     I_test = utils.make_grid(image, nrow=8, normalize=True, scale_each=True)
#     #writer.add_image('test/image', I_test, i)
#     filename = fileNames[i]
#     if a1 is not None:
#         attn1 = visualize_attn(I_test, a1, up_factor=8, nrow=8)
#         #print(attn1.shape)
#         torchvision.utils.save_image(attn1, os.path.join("adversarial_result/vis",filename+"_attn1.png"))
#         #writer.add_image('test/attention_map_1', attn1, i)
#     if a2 is not None:
#         attn2 = visualize_attn(I_test, a2, up_factor=2*8, nrow=8)
#         torchvision.utils.save_image(attn2, os.path.join("adversarial_result/vis",filename+"_attn2.png"))


#print(torch.softmax(predict,1))
## visualize saliency map
# target_classes = [1,1,1]
# filenames = ["GAP","PGD","ori"]
# for i in range(3):
#     filename = filenames[i]
#     image = images[i]
#     image = image.unsqueeze(0)
#     image.requires_grad = True
#     target_class = target_classes[i]
#     output,_,_ = pretrained_clf(image)
#     one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
#     one_hot_output[0][target_class] = 1
#     one_hot_output = one_hot_output.cuda()
#     # Backward pass
#     output.backward(gradient=one_hot_output)
#     print(image.grad.data.shape)
#     gradient_arr = image.grad.data.cpu().numpy()[0]
#     gradient_arr_grey = libadver.visutils.convert_to_grayscale(gradient_arr)
#     print(gradient_arr_grey.shape)
#     gradient_arr_grey = 1 - gradient_arr_grey
#     libadver.visutils.save_gradient_images(gradient_arr_grey, os.path.join("adversarial_result/vis",filename+"_saliency.png"))


## visulize feature map
net = AttnVGG(num_classes=2, attention=True, normalize_attn=True, vis = True)
# net = VGG(num_classes=2, gap=False)
checkpoint = torch.load(modelFile)
net.load_state_dict(checkpoint['state_dict'])
pretrained_clf = nn.DataParallel(net).cuda()
pretrained_clf.eval()

for i in range(3):
    image = images[i]
    image = image.unsqueeze(0)
    #print(len(output))
    output,_,_,block1,block2,block3,block4,block5,g_hat = pretrained_clf(image)
    print(output)
    for j in range(block2.shape[1]):
        activation = block2[0,j,:,:]
        activation = activation.cpu().detach().numpy()
        heatmap_activation = libadver.visutils.generate_colormap(activation,'jet')
        heatmap_activation = heatmap_activation.resize([224,224])
        heatmap_activation.save("adversarial_result/vis/feature/%d_%d.png" %(i,j))
