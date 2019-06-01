import libadver.attack as attack
import libadver
import net
import torch.backends.cudnn as cudnn
from torchvision import  transforms
import torchvision
import torch
import numpy as np
from data import load_data
import os
import torch.nn as nn
import libadver.models.generators as generators
import torch.optim as optim


pretrained_clf = net.AttnVGG(num_classes=2, attention=True, normalize_attn=True, dropout = 0.8)
pretrained_clf = torch.nn.DataParallel(pretrained_clf)
cudnn.benchmark = True

model_file = "messidor_attVGG_latest.pkl"
pretrained_clf.load_state_dict(torch.load(model_file))
pretrained_clf = pretrained_clf.cuda()
pretrained_clf.eval()

print("=====> Load Data")
testRootDir = "/store/dataset/messidor/binary_classification/test"
trainRootDir = "/store/dataset/messidor/binary_classification/train"
testBatchSize = 16
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])
testset = torchvision.datasets.ImageFolder(testRootDir,transform=test_transforms)
testloader = torch.utils.data.DataLoader(testset,batch_size=testBatchSize,shuffle=True,drop_last=False,num_workers=4)

trainset = torchvision.datasets.ImageFolder(trainRootDir,transform=test_transforms)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=testBatchSize,shuffle=True,drop_last=False,num_workers=4)


print(len(trainloader))

isTrain = False
isTestDataset = True

params = {
        "attackModelPath" : None,
        "mag_in" : 15.0,
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
saveModelPath = "adversarial_result/GAP/GAP_im_m5n2.pth"
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

        for i, data in enumerate(testloader):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            adv_images = GAPAttack.generate(images)
            predicted,_,_ = pretrained_clf(adv_images)
            predicted_labels = torch.argmax(predicted,1)
            #print(predicted_labels)
            correct += torch.sum(predicted_labels.eq(labels))
            #print(targets)
            total += images.shape[0]
            print("ACC:%.3f | %d,%d" %(100.0*float(correct) / total, correct, total))

            delta_ims = adv_images - images
            delta_im_temp = torch.zeros(images.size())

            for c2 in range(3):
                adv_images.data[:,c2,:,:] = (adv_images.data[:,c2,:,:] * std[c2]) + mean[c2]
                images.data[:,c2,:,:] = (images.data[:,c2,:,:] * std[c2]) + mean[c2]
                delta_im_temp.data[:,c2,:,:] = (delta_ims.data[:,c2,:,:] * std[c2]) + mean[c2]
            #delta_ims = adv_images - images
            #print(delta_ims.abs().max() * 255.0)
            #torchvision.utils.save_image(delta_ims,"delta_im.png")
            torchvision.utils.save_image(adv_images.data, 'reconstructed.png')
            torchvision.utils.save_image(images.data, 'original.png')
            torchvision.utils.save_image(delta_im_temp.data, 'delta_im.png')
            #break

    else:
        images, labels = load_data(isBenign = False, transform = transform)
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

        adv_image = adv_images[4]
        adv_image_PIL = libadver.visutils.recreate_image(adv_image,mean,std)
        libadver.visutils.save_image(adv_image_PIL,"adversarial_result/GAP/malignant/adv_img_4.png")

        #delta_im = delta_ims[0].data.numpy()
        delta_im = delta_ims[4]
        #libadver.visutils.save_gradient_images(delta_im,"adversarial_result/GAP/benign/delta_im_0.png")
        delta_im_PIL = libadver.visutils.recreate_image(delta_im,mean,std)
        libadver.visutils.save_image(delta_im_PIL,"adversarial_result/GAP/malignant/delta_im_4.png")
