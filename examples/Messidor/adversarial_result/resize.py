from PIL import Image
import os
sourceRoot = "ori_img"
desRoot = "pre_img"
cats = ["0","1"]


for cat in cats:
    sourceDir = os.path.join(sourceRoot,cat)
    desDir = os.path.join(desRoot, cat)

    filenames = os.listdir(sourceDir)
    filenames.sort()
    for i,filename in enumerate(filenames):
        filePath = os.path.join(sourceDir, filename)
        img = Image.open(filePath)
        img = img.resize([224,224])
        img.save(os.path.join(desDir, "ori_img_%d.png" %i ))
