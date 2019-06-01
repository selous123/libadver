from PIL import Image
import os
from libadver import visutils


# benignImg = benignImgs[0]
# benignPath = os.path.join(benignRoot, benignImg)
# img = Image.open(benignPath)
# text = "Benign:98.91%"
# img_text = visutils.draw_text(img,text,'green',length=160)
# img_text.save(os.path.join(benignRoot, "ori_img_0_text.png"))
#
# advImg = benignGAPImgs[0]
# advPath = os.path.join(benignGAPRoot, advImg)
# img = Image.open(advPath)
# text = "Malignant:85.69%"
# img_text = visutils.draw_text(img,text,'red',length=180)
# img_text.save(os.path.join(benignGAPRoot, "adv_img_0_text.png"))

## malign Img with PGD attack
malignantRoot = "./adversarial_result/pre_img/1"
malignImg = "ori_img_0.png"
malignPath = os.path.join(malignantRoot, malignImg)
img = Image.open(malignPath)
text = "DR:98.72%"
img_text = visutils.draw_text(img,text,'green',length=180)
img_text.save(os.path.join(malignantRoot, "ori_img_0_text.png"))

malignantAdvRoot = "./adversarial_result/PGD/1"
malignAdvImg = "adv_img_0.png"
malignAdvPath = os.path.join(malignantAdvRoot, malignAdvImg)
advImg = Image.open(malignAdvPath)
text = "No DR:100%"
advImg_text = visutils.draw_text(advImg,text,'red',length=180)
advImg_text.save(os.path.join(malignantAdvRoot, "adv_img_0_text.png"))
