import torch.nn as nn
class DeepFool():
    def __init__(self, model):
        pass


    def generate(self, inputs):
        x = inputs.detach()

        x.requires_grad_()
        logits = self.model(x)

        ## ind with shape [1, n_classes]
        ind = torch.sort(logits, dim=1)
        label = ind[0]


        for k in range(1, num_classes):
            logit = logits[0, ind[k]]
            logit.backward(retain_graph = True)
            x.data.grad

    def parse_params(self):
        pass



if __name__=="__main__":

    import torchvision.models as models
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    pretrained_clf = models.resnet18(pretrained = True)
    pretrained_clf = pretrained_clf.cuda()
    pretrained_clf.eval()

    im_orig = Image.open('../models/cat_dog.png')

    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]


    # Remove the mean
    im = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std)])(im_orig)

    print(im.shape)

    im.unsqueeze_(0)
    im = im.cuda()
    im.requires_grad_()
    output = pretrained_clf(im)
    output[0,0].backward()
    print(im.grad.shape)
    print(torch.argmax(output, dim=1))

    #trainDataset = torchvision.dataset.CIFAR10()
