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


pgd_params = {
            'ord': np.inf,
            'y': None,
            'eps': 5.0 / 255,
            'eps_iter': 2.5 / 255,
            'nb_iter': 5,
            'rand_init': True,
            'rand_minmax': 5.0 / 255,
            'clip_min': 0.,
            'clip_max': 1.,
            'sanity_checks': True
        }
PGDAttack = attack.ProjectGradientDescent(model = pretrained_clf)
correct = 0
total = 0
isAttackDataset = False
if isAttackDataset is True:
    for image,label in testloader:
        image, label = image.cuda(), label.cuda()
        ## non targeted
        pgd_params['y'] = label
        pgd_params['clip_min'] = torch.min(image)
        pgd_params['clip_max'] = torch.max(image)

        adv_x = PGDAttack.generate(image, **pgd_params)
        #adv_x = image
        outputs, _, _ = pretrained_clf(adv_x)
        pred_adv = torch.argmax(outputs, dim = 1)

        # for c in range(3):
        #     adv_x.data[:,c,:,:] = (adv_x.data[:,c,:,:] * std[c]) + mean[c]
        #     image.data[:,c,:,:] = (image.data[:,c,:,:] * std[c]) + mean[c]
        # torchvision.utils.save_image(adv_x, "adv.png", nrow = 4)
        # torchvision.utils.save_image(image, "image.png", nrow = 4)

        total = total + image.size(0)
        correct = correct + label.eq(pred_adv).sum()
        print("ACC: %.4f (%d, %d)" %(float(correct) / total, correct, total))
else:
    ## generate adversarial sample by single sample
    file_dir = "/home/lrh/project/myPackage/libadver/examples/Messidor/adversarial_result/ori_img/1"
    file_names = os.listdir(file_dir)
    file_names.sort()
    file_paths = [os.path.join(file_dir,file_name) for file_name in file_names]
    images = load_data(file_paths,test_transforms)

    pgd_params['y'] = torch.LongTensor(len(file_names)).zero_().cuda() + 1
    pgd_params['clip_min'] = torch.min(images)
    pgd_params['clip_max'] = torch.max(images)
    #print(pgd_params['y'])
    adv_x = PGDAttack.generate(images, **pgd_params)
    outputs, _, _ = pretrained_clf(adv_x)
    #pred_adv = torch.argmax(outputs, dim = 1)
    print(torch.softmax(outputs,dim=1))
    #print(pred_adv)

    adv_images = adv_x.cpu()
    images = images.cpu()
    delta_ims = (adv_images - images).cpu()




    #image_PIL = libadver.visutils.recreate_image(image,mean,std)
    #libadver.visutils.save_image(image_PIL,"adversarial_result/pre_img/malignant/ori_img_%d.png" %i)
    for i in range(len(file_names)):
        adv_image = adv_images[i]
        delta_im = delta_ims[i]

        adv_image_np = libadver.visutils.recreate_image(adv_image,mean,std)
        libadver.visutils.save_image(adv_image_np,"adversarial_result/PGD/1/adv_img_%d.png" %i)

        delta_im = delta_ims[i].data.numpy()
        libadver.visutils.save_gradient_images(delta_im,"adversarial_result/PGD/1/delta_im_%d.png" %i)
        #delta_im_np = libadver.visutils.recreate_image(delta_im,mean=(0,0,0),std=(1,1,1))
        #libadver.visutils.save_image(delta_im_np,"adversarial_result/PGD/1/delta_im_PGD_%d.png" %i)

    #outputs, _, _ = pretrained_clf(images)
    #print(torch.softmax(outputs,dim=1))
