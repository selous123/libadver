import torch

class BenchmarkGenerator():
    """
    To generate common corruptions and perturbations images for Benchmarking
    Paper link (Hendrycks et al. 2019) : https://openreview.net/forum?id=HJz6tiCqYm
    """
    def __init__(self, **kwargs):
        self.parse_params(**kwargs)


    def generate(self, inputs):
    """
    Generate function parameters:
        :param inputs: input image with shape [in_channel, height, width]
    Return:
        output the corruptions and perturbations images with shape [None, in_channel, height, width]
    """
        pass

    def parse_params(self):
        pass

if __name__=="__main__":
    pass
