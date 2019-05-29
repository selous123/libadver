from PIL import Image
import os
from libadver import visutils

malignantRoot = "./adversarial_result/pre_img/malignant"
malignImgs = [
    "ori_img_0.png", "ori_img_1.png", "ori_img_2.png",
    "ori_img_3.png", "ori_img_4.png"
]

benignRoot = "./adversarial_result/pre_img/benign"
benignImgs = [
    "ori_img_0.png", "ori_img_1.png", "ori_img_2.png",
    "ori_img_3.png", "ori_img_4.png"
]

malignantAdvRoot = "./adversarial_result/PGD/malignant"
malignAdvImgs = [
    "adv_img_0.png", "adv_img_1.png", "adv_img_2.png",
    "adv_img_3.png", "adv_img_4.png"
]

benignGAPRoot = "./adversarial_result/GAP/benign"
benignGAPImgs = [
    "adv_img_0.png", "adv_img_1.png", "adv_img_2.png",
    "adv_img_3.png", "adv_img_4.png"
]

benignImg = benignImgs[0]
benignPath = os.path.join(benignRoot, benignImg)
img = Image.open(benignPath)
text = "Benign:98.91%"
img_text = visutils.draw_text(img,text,'green',length=160)
img_text.save(os.path.join(benignRoot, "ori_img_0_text.png"))

advImg = benignGAPImgs[0]
advPath = os.path.join(benignGAPRoot, advImg)
img = Image.open(advPath)
text = "Malignant:85.69%"
img_text = visutils.draw_text(img,text,'red',length=180)
img_text.save(os.path.join(benignGAPRoot, "adv_img_0_text.png"))

## malign Img with PGD attack
# malignImg = malignImgs[0]
# malignPath = os.path.join(malignantRoot, malignImg)
# img = Image.open(malignPath)
# text = "Malignant:98.93%"
# img_text = visutils.draw_text(img,text,'green',length=180)
# img_text.save(os.path.join(malignantRoot, "ori_img_0_text.png"))

# malignAdvImg = malignAdvImgs[0]
# malignAdvPath = os.path.join(malignantAdvRoot, malignAdvImg)
# advImg = Image.open(malignAdvPath)
# text = "Benign:100%"
# advImg_text = visutils.draw_text(advImg,text,'red',length=180)
# advImg_text.save(os.path.join(malignantAdvRoot, "adv_img_0_text.png"))
