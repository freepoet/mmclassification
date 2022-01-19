import torch
# torch.cuda.set_device(0)
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage()   #0-1 转 0-255
import cv2
import os
import argparse

import ipdb

def parse_args():

    parser = argparse.ArgumentParser(description='mnist2png_with_scaled_height')
    parser.add_argument('--mnist_path', help=' raw mnist_path')
    parser.add_argument('--png_path_prefix', help='png_path_prefix')
    parser.add_argument('--scale_ratio', type=float, help='height scale_ratio')
    args = parser.parse_args()
    return args

def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[31m%s\033[0m" % '   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent * 100) + end_str
    print(bar, end='', flush=True)



def mnist2png(mnist_path,png_path_prefix,scale_ratio):


    # 读取MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),  # 0-255  转为  0-1
        transforms.Normalize((0.5,), (0.5,))  # 0-1  转为  -1-1
    ])
    trainset = torchvision.datasets.MNIST(
        root=mnist_path,
        train=True,
        download=False,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        shuffle=False,          # shuffle must be False ,otherwise
        num_workers=0
    )
    testset = torchvision.datasets.MNIST(
        root=mnist_path,
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
    for i, data in enumerate(trainloader,0):
        images, labels = data
        img=(images.squeeze(0)).squeeze(0).numpy()
        width=img.shape[0]
        height=img.shape[1]
        img_length=width*height
        new_width=width
        new_height=round(height*scale_ratio)

        img_new=cv2.resize(img,(new_width,new_height),interpolation=cv2.INTER_LINEAR)
        save_path = os.path.join(png_path_prefix,'ratio_'+str(scale_ratio),'train-images',str(int(labels)))
        # cv2.imwrite() will not write an image in another directory if the directory does not exist.
        # You first need to create the directory before attempting to write to it:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path+f'/{i}.png',(img_new+1)/2*255)
        start_str='Converting Trainingset: '
        end_str = '100%'
        process_bar(i / 60000, start_str=start_str, end_str=end_str, total_length=15)

    for i, data in enumerate(testloader,0):
        images, labels = data
        img=(images.squeeze(0)).squeeze(0).numpy()
        width=img.shape[0]
        height=img.shape[1]
        img_length=width*height

        new_width=width
        new_height=round(height*scale_ratio)

        img_new=cv2.resize(img,(new_width,new_height),interpolation=cv2.INTER_LINEAR)
        save_path = os.path.join(png_path_prefix,'ratio_'+str(scale_ratio),'test-images',str(int(labels)))
        # cv2.imwrite() will not write an image in another directory if the directory does not exist.
        # You first need to create the directory before attempting to write to it:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path+f'/{i}.png',(img_new+1)/2*255 )
        start_str='Converting Testingset: '
        end_str = '100%'
        process_bar(i / 10000, start_str=start_str, end_str=end_str, total_length=15)
        print('\n')

def main():
    args = parse_args()
    if args.mnist_path is not None:
        mnist_path=args.mnist_path

    if args.png_path_prefix is not None:
        png_path_prefix=args.png_path_prefix

    if args.scale_ratio is not None:
        scale_ratio=args.scale_ratio

    # mnist_path=/media/n/SanDiskSSD/HardDisk/data/
    # png_path_prefix==/media/n/SanDiskSSD/HardDisk/data/MNIST/Visualized
    # scale_ratio = 1.0  (int)
    mnist2png(mnist_path,png_path_prefix,scale_ratio)

if __name__ == '__main__':
    main()




#python custom/configs/my_own/mnist2png_with_scaled_height.py --mnist_path /media/n/SanDiskSSD/HardDisk/data/ --png_path_prefix /media/n/SanDiskSSD/HardDisk/data/MNIST/Visualized --scale_ratio 1.0