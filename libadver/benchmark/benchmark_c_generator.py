# import torch
# import os
# from PIL import Image
# import os.path
# import time
# import torch
# import torchvision.datasets as dset
# import torchvision.transforms as trn
# import torch.utils.data as data
# import numpy as np
from libadver.benchmark.utils import *
#from utils import *
# #from utils import *
#
# from PIL import Image



class Benchmark_C_Generator():
    def __init__(self, transform=None, target_transform=None, loader=default_loader, **kwargs):
        self.parse_params(**kwargs)
        classes, class_to_idx = find_classes(self.dataroot)
        imgs = make_dataset(self.dataroot, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + self.dataroot + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

            if self.method == "gaussian_noise":
                img = gaussian_noise(img, self.severity)
            elif self.method == "shot_noise":
                img = shot_noise(img, self.severity)
            elif self.method == "impulse_noise":
                img = impulse_noise(img, self.severity)
            elif self.method == "defocus_blur":
                img = defocus_blur(img, self.severity)
            elif self.method == "glass_blur":
                img = glass_blur(img, self.severity)
            elif self.method == "motion_blur":
                img = motion_blur(img, self.severity)
            elif self.method == "zoom_blur":
                img = zoom_blur(img, self.severity)
            elif self.method == "snow":
                img =  snow(img, self.severity)
            elif self.method == "frost":
                img = frost(img, self.severity)
            elif self.method == "fog":
                img = fog(img, self.severity)
            elif self.method == "brightness":
                img = brightness(img, self.severity)
            elif self.method == "contrast":
                img = contrast(img, self.severity)
            elif self.method == "elastic_transform":
                img = elastic_transform(img, self.severity)
            elif self.method == "pixelate":
                img = pixelate(img, self.severity)
            elif self.method == "jpeg_compression":
                img = jpeg_compression(img, self.severity)
            elif self.method == "speckle_noise":
                img = speckle_noise(img, self.severity)
            elif self.method == "gaussian_blur":
                img = gaussian_blur(img, self.severity)
            elif self.method == "saturate":
                img = saturate(img, self.severity)



        if self.target_transform is not None:
            target = self.target_transform(target)

        save_path = self.saveroot + self.method + \
                    '/' + str(self.severity) + '/' + self.idx_to_class[target]

        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except OSError:
                pass
#            os.makedirs(save_path)

        save_path += path[path.rindex('/'):]

        Image.fromarray(np.uint8(img)).save(save_path, quality=85, optimize=True)

        return 0  # we do not care about returning the data

    def __len__(self):
        return len(self.imgs)

    def parse_params(self,
                     dataroot = None,
                     saveroot = None,
                     method = "Gaussian Noise",
                     severity = 0,
                     **kwargs):
        self.dataroot = dataroot
        self.saveroot = saveroot
        self.method = method
        self.severity = severity

        if self.dataroot is not None and not os.path.exists(self.dataroot):
            raise FileNotFoundError("%s file is not exists" %self.dataroot)

        if self.method not in ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression", "speckle_noise", "gaussian_blur", "saturate"]:
            raise ValueError("method must be either\" gaussian_noise \",\"  shot_noise \",\" impulse_noise \",\" defocus_blur \",\" glass_blur \",\" motion_blur \",\" zoom_blur \",\" snow \",\" frost \",\" fog \",\" brightness \",\" contrast \",\" elastic_transform \",\" pixelate \",\" jpeg_compression \",\" speckle_noise \",\" gaussian_blur \",\" spatter \",\" saturate \".")

        if severity not in [1,2,3,4,5]:
            raise ValueError("severity must be either\" 1 \",\" 2 \",\" 3 \",\" 4 \",\" 5 \".")

if __name__=="__main__":
    # import torch
    # import os
    # from PIL import Image
    # import os.path
    # import time
    # import torch
    # import torchvision.datasets as dset
    # import torchvision.transforms as trn
    # import torch.utils.data as data
    # import numpy as np
    # from libadver.defence.utils import *
    # #from utils import *
    #
    # from PIL import Image
#    import libadver.defence as defence
    import cv2
    import torch
    import torchvision.transforms as transforms

    dataroot = "/home/lrh/dataset/ISIC_data_2016/Val_robustness"
    saveroot = "/home/lrh/dataset/ISIC_data_2016/Val_robustness_C/"

    params = {
        "method" : "frost",
        "severity" : 2,
    }
    params["dataroot"] = dataroot
    params["saveroot"] = saveroot
    print(params)

    print("making Benchmark_C.....")
    Benchmark_C_dataset = Benchmark_C_Generator(transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)]),**params)
    Benchmark_C_dataset_loader = torch.utils.data.DataLoader(Benchmark_C_dataset, batch_size=10, shuffle=False, num_workers=4)

    for _ in Benchmark_C_dataset_loader: continue
    print("\ndone")
