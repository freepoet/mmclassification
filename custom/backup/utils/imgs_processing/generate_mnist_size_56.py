import torch
# torch.cuda.set_device(0)
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage()   #0-1 转 0-255
import cv2
import os
from PIL import Image
from imgs_zero_padding import my_img_pad
import ipdb
import numpy as np


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print
       # "---  new folder...  ---"
        print
       # "---  OK  ---"

    else:
        print
       # "---  There is this folder!  ---"
# 读取MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 0-255  转为  0-1
    transforms.Normalize((0.5,), (0.5,))  # 0-1  转为  -1-1
])
trainset = torchvision.datasets.MNIST(
    root='/media/n/SanDiskSSD/HardDisk/data',
    train=True,
    download=False,
    transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=1,  #batch_size should be larger than 4, otherwaise, loss = criterion(outputs, labels) will report a error.

    shuffle=False,
    num_workers=0
)
testset = torchvision.datasets.MNIST(
    root='/media/n/SanDiskSSD/HardDisk/data',
    train=False,
    download=False,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)
### generate 56*56imgs
# save_path='/media/n/SanDiskSSD/HardDisk/data/MNIST_Visualized/size_56'
# single_size = 56
# for i in range(10):
#     mkdir(save_path+f'/train/{i}')
#     mkdir(save_path+f'/test/{i}')
# for i, data in enumerate(testloader,0):
#     images, labels = data
#     img=(images.squeeze(0)).squeeze(0).numpy()
#     img_pil = Image.fromarray(img)
#     img = my_img_pad(img_pil, single_size)
#     # ipdb.set_trace()
#     cv2.imwrite(save_path+f'/test/{int(labels)}/{i}.png',img*255)

### save imgs to test.npy file
dataset_dir='/media/n/SanDiskSSD/HardDisk/data/MNIST_Visualized/size_56/test'
labels=os.listdir(dataset_dir)
img_old=np.zeros((1,56,56))
gt_label=[]
for label in labels:
    img_dir = os.path.join(dataset_dir,label)
    img_files=os.listdir(img_dir)
    for img_file in img_files:
        img_new = cv2.imread(os.path.join(img_dir, img_file), 0)  # 56 56
        img_new = img_new[np.newaxis, :]
        img_new = np.vstack((img_old, img_new))
        img_old = img_new
        gt_label.append(int(label))
    print('generatd ok!')
img_old = np.delete(img_old, 0, axis=0)
gt_label=np.array(gt_label)
np.save('npy_file/test_img.npy',img_old)
np.save('npy_file/test_label.npy',gt_label)

#### save imgs to trian.npy file  train文件太大 分开存储
# dataset_dir='/media/n/SanDiskSSD/HardDisk/data/MNIST_Visualized/size_56/train'
# labels=os.listdir(dataset_dir)
# for label in labels:
#     img_old = np.zeros((1, 56, 56))
#     gt_label = []
#
#     img_dir = os.path.join(dataset_dir,label)
#     img_files=os.listdir(img_dir)
#     for img_file in img_files:
#         img_new = cv2.imread(os.path.join(img_dir, img_file), 0)  # 56 56
#         img_new = img_new[np.newaxis, :]
#         img_new = np.vstack((img_old, img_new))
#         img_old = img_new
#         gt_label.append(int(label))
#     print('generatd ok!')
#     img_old = np.delete(img_old, 0, axis=0)
#     gt_label=np.array(gt_label)
#     np.save('npy_file/train_img_'+f'{int(label)}.npy',img_old)
#     np.save('npy_file/train_label_'+f'{int(label)}.npy',gt_label)
