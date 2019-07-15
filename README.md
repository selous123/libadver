# libadver
Package for adversarial attack in pytorch

## Installation

We developed libadver under Python 3.6 and PyTorch 1.0.0 & 0.4.1. To install libadver, simply run

### 1.clone repo

```
git clone https://github.com/selous123/libadver.git
```

### 2.install

```
python setup.py install
&&
pip install .
//pip install -e . ##editable mode
```

## Examples
test PGD Attack
```
import numpy as np
import torch
pgd_params = {
            'ord': np.inf,
            'y': None,
            'eps': 16.0 / 255,
            'eps_iter': 2.55 / 255,
            'nb_iter': 40,
            'rand_init': True,
            'rand_minmax': 16.0 / 255,
            'clip_min': 0.,
            'clip_max': 1.,
            'sanity_checks': True
        }


import libadver.attack as attack

PGDAttack = attack.ProjectGradientDescent(model = pretrained_clf)

correct = 0
total = 0
for image,label in testloader:
    image, label = image.cuda(), label.cuda()
    ## non targeted
    pgd_params['y'] = label
    pgd_params['clip_min'] = torch.min(image) 
    pgd_params['clip_max'] = torch.max(image)
    
    adv_x = PGDAttack.generate(image, **pgd_params)

    outputs, _, _ = pretrained_clf(adv_x)
    pred_adv = torch.argmax(outputs, dim = 1)
    
    for c in range(3):
        adv_x.data[:,c,:,:] = (adv_x.data[:,c,:,:] * std[c]) + mean[c]
        image.data[:,c,:,:] = (image.data[:,c,:,:] * std[c]) + mean[c]
    torchvision.utils.save_image(adv_x, "adv.jpg", nrow = 4)
    torchvision.utils.save_image(image, "image.jpg", nrow = 4)

    total = total + image.size(0)
    correct = correct + label.eq(pred_adv).sum()
    print("ACC: %.4f (%d, %d)" %(float(correct) / total, correct, total))
```

For runable examples, see from [projected_gradient_descent.py]()


## List of Attack
1.FGSM

2.PGD

3.deepfool

4.universal adversarial

5.generative adversarial

6....

## Benchmark for robustness

### 1.Benchmark_C

The method for creating benchmark_c must be either "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression", "speckle_noise", "gaussian_blur", "saturate".

The severity of noise must be either 1,2,3,4,5

examples

```
import libadver.benchmark as benchmark
import torch
import torchvision.transforms as transforms

dataroot = "/home/lrh/dataset/ISIC_data_2016/robustness"
saveroot = "/home/lrh/dataset/ISIC_data_2016/robustness_c_test/"

params = {
    "method" : "snow",
    "severity" : 2,
}
params["dataroot"] = dataroot
params["saveroot"] = saveroot
print(params)

print("making Benchmark_C.....")
Benchmark_C_dataset = benchmark.Benchmark_C_Generator(transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)]),**params)
Benchmark_C_dataset_loader = torch.utils.data.DataLoader(Benchmark_C_dataset, batch_size=10, shuffle=False, num_workers=4)

for _ in Benchmark_C_dataset_loader: continue
print("\ndone")
```

### 2.Benchmark_P

The method for creating benchmark_p must be either "gaussian_noise", "shot_noise", "motion_blur", "zoom_blur", "snow", "brightness", "rotate","tilt","scale","translate" ,"spatter","speckle_noise", "gaussian_blur", "saturate","shear".

The frameNum for benchmark_p must be a positive integer.

examples

```
import libadver.benchmark as benchmark
params = {
        "dataroot" : "/home/lrh/dataset/ISIC_data_2016/robustness/",
        "saveroot" : "/home/lrh/dataset/ISIC_data_2016/robustness_p",
        "method" : "shear",
        "frameNum": 5
}


print("making Benchmark_P.....")
benchmark.Benchmark_P_Generator(**params)

print("\ndone")
```


## Comming soon
1. defence

## License 

This project is licensed under the GNU General Public License v3.0. The terms and conditions can be found in the LICENSE files.

## Contribution
Tao Zhang (lrhselous@nuaa.edu.cn)

Mengting Xu