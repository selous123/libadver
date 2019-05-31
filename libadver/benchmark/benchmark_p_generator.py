from libadver.benchmark.utils import *
#from utils import *

class Benchmark_P_Generator():
    def __init__(self,**kwargs):
        self.parse_params(**kwargs)
        print(self.method)
        print("the frameNum is %s." %self.frameNum)
        if self.method == "shot_noise":
            shot_noise_p(self.dataroot, self.saveroot, self.frameNum)
        if self.method == "gaussian_noise":
            gaussian_noise_p(self.dataroot, self.saveroot, self.frameNum)
        if self.method == "motion_blur":
            motion_blur_p(self.dataroot, self.saveroot, self.frameNum)
        if self.method == "zoom_blur":
            zoom_blur_p(self.dataroot, self.saveroot, self.frameNum)
        if self.method == "snow":
            snow_p(self.dataroot, self.saveroot, self.frameNum)
        if self.method == "brightness":
            brightness_p(self.dataroot, self.saveroot, self.frameNum)
        if self.method == "translate":
            translate_p(self.dataroot, self.saveroot, self.frameNum)
        if self.method == "rotate":
            rotate_p(self.dataroot, self.saveroot, self.frameNum)
        if self.method == "tilt":
            tilt_p(self.dataroot, self.saveroot, self.frameNum)
        if self.method == "scale":
            scale_p(self.dataroot, self.saveroot, self.frameNum)
        if self.method == "speckle_noise":
            speckle_noise_p(self.dataroot, self.saveroot, self.frameNum)
        if self.method == "gaussian_blur":
            gaussian_blur_p(self.dataroot, self.saveroot, self.frameNum)
        if self.method == "spatter":
            spatter_p(self.dataroot, self.saveroot, self.frameNum)
        if self.method == "shear":
            shear_p(self.dataroot, self.saveroot, self.frameNum)


    def parse_params(self,
                     dataroot = None,
                     saveroot = None,
                     method = "shot_noise",
                     frameNum = 6,
                     **kwargs):
        self.dataroot = dataroot
        self.saveroot = saveroot
        self.method = method
        self.frameNum = frameNum

        if self.dataroot is not None and not os.path.exists(self.dataroot):
            raise FileNotFoundError("%s file is not exists" %self.dataroot)

        if self.method not in ["gaussian_noise", "shot_noise", "motion_blur", "zoom_blur", "snow", "brightness", "rotate","tilt","scale","translate" ,"spatter","speckle_noise", "gaussian_blur", "saturate","shear"]:
            raise ValueError("method must be either \"gaussian_noise\", \"shot_noise\", \"motion_blur\", \"zoom_blur\", \"snow\", \"brightness\", \"rotate\", \"snow\", \"tilt\", \"scale\", \"translate\", \"spatter\", \"speckle_noise\", \"gaussian_blur\", \"saturate\", \"shear\".")


if __name__=="__main__":

    import cv2
    import torch
    import torchvision.transforms as transforms

    params = {
            "dataroot" : "/home/lrh/dataset/ISIC_data_2016/robustness/",
            "saveroot" : "/home/lrh/dataset/ISIC_data_2016/robustness_p",
            "method" : "shear",
            "frameNum": 5
    }


    print("making Benchmark_P.....")
    Benchmark_P_Generator(**params)

    print("\ndone")
