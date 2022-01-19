import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2



filepath = r'/media/n/SanDiskSSD/HardDisk/data/MNIST_Visualized/size_56/train'  # 数据集目录
labels = os.listdir(filepath)
size=56*56
num=0
R_channel = 0
for label in labels:
    imgs = os.listdir(os.path.join(filepath,label))
    for img in imgs:
        img_file=cv2.imread(os.path.join(filepath,label,img),0)
        R_channel = R_channel + np.sum(img_file/255)/size
        num=num+1
R_mean = R_channel / num

num=0
R_channel=0
for label in labels:
    imgs = os.listdir(os.path.join(filepath, label))
    for img in imgs:
        img_file = cv2.imread(os.path.join(filepath, label, img), 0)
        R_channel = R_channel + np.sum((img_file/255 - R_mean) ** 2)/size
        num = num + 1
R_std = np.sqrt(R_channel/ num)

print("mean is %f" % R_mean)
print("std is %f" % R_std)