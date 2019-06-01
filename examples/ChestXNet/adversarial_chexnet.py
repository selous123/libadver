import libadver.attack as attack
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from PIL import Image
import torch
import re
import os
import sys
sys.path.append("..")
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score

def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    N_CLASSES = gt_np.shape[1]
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

CKPT_PATH = '../model.pth.tar'
N_CLASSES = 14
cudnn.benchmark = True
# initialize and load the model
model = DenseNet121(N_CLASSES).cuda()
model = torch.nn.DataParallel(model).cuda()
model.eval()
if os.path.isfile(CKPT_PATH):
    print("=> loading checkpoint")
    checkpoint = torch.load(CKPT_PATH)

    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    state_dict = checkpoint['state_dict']
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    model.load_state_dict(state_dict)
    print("=> loaded checkpoint")
else:
    print("=> no checkpoint found")


DATA_DIR = '/store/dataset/ChestXray-NIHCC/images_v1_small'
TEST_IMAGE_LIST = '../ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 10
#IMAGE_PATH = '/store/dataset/ChestXray-NIHCC/images/00017046_015.png'
#img = Image.open(IMAGE_PATH).convert('RGB')
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                image_list_file=TEST_IMAGE_LIST,
                                transform = test_transform
                                )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=8, pin_memory=True)

#img_torch = transform(img)
#img_torch.shape


pgd_params = {
    'ord': np.inf,
    'y': None,
    'y_target': None,
    'eps': 5.0 / 255,
    'eps_iter': 0.5 / 255,
    'nb_iter': 40,
    'rand_init': True,
    'rand_minmax': 5.0 / 255,
    'clip_min': 0.,
    'clip_max': 1.,
    'sanity_checks': True,
    'criterion' : nn.BCELoss()
}

pgdAttack = attack.ProjectGradientDescent(model)

# store the results
gt = torch.FloatTensor()
pred = torch.FloatTensor()
pred_advx = torch.FloatTensor()


for batchIdx, (images, labels) in enumerate(test_loader):
    print("[%d|%d]" %(batchIdx,len(test_loader)))
    images, labels = images.cuda(), labels.cuda()
    pgd_params["clip_min"] = torch.min(images)
    pgd_params["clip_max"] = torch.max(images)
    #gt = torch.FloatTensor([[0,0,0,0,0,0,0,0,0,0,0,0,1,0]]).cuda()
    pgd_params["y"] = labels
    adv_x = pgdAttack.generate(images, **pgd_params)
    #torchvision.utils.save_image(adv_x, "adv_x.png", normalize = True)
    #torchvision.utils.save_image(images, "img.png", normalize = True)
    #torchvision.utils.save_image(adv_x - images, "delta_m.png", normalize = True)
    x_pred = model(images)
    adv_pred = model(adv_x)
    gt = torch.cat((gt, labels.detach().cpu()), 0)
    pred = torch.cat((pred, x_pred.detach().cpu()), 0)
    pred_advx = torch.cat((pred_advx, adv_pred.detach().cpu()),0)
    # print(x_pred)
    # print(adv_pred)
    #break

gt_np = gt.numpy()
pred_np = pred.numpy()
pred_advx_np = pred_advx.numpy()
np.save("gt.npy", gt_np)
np.save("pred.npy", pred_np)
np.save("pred_advx.npy", pred_advx_np)
