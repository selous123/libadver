import torch.nn as nn

class GenerativeAdversarialPerturbations():
    def __init__(self, model, attackModel, **args):
        if not isinstance(model, nn.Module):
            raise TypeError("The model argument should be the instance of"
                            "torch.nn.Module")
        self.model = model
        self.attackModel = attackModel



    def train(self, saveModelPath):

        self.attackModelPath = saveModelPath

    def generate(self, inputs):
        ## assure that attack model has been trained before
        if self.attackModelPath is None:
            raise ValueError("Generate function is should be invoked"
                        "before training")
        x = input.detach()

        return x

    def parse_params(self,
                    attackModelPath = None,
                    ord = "inf",
                    eps = 0.3):
        self.attackModelPath = attackModelPath
        pass



if __name__ == "__main__":
    params = {
        attackModelPath : None,
    }
