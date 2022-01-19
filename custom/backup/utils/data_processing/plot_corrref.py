# # Autocorrelation coefficient
# import cv2
# import os
# import argparse
# # def parse_args():
# #
# #     parser = argparse.ArgumentParser(description='mnist2png_with_scaled_height')
# #     parser.add_argument('--img_path', help='absolute path')
# #     parser.add_argument('--ratio_min', type=float)
# #     parser.add_argument('--ratio_max', type=float)
# #     args = parser.parse_args()
# #     return args
#
# def auto_correlation(img_path,ratio_min,ratio_max):
#     dirs = os.listdir(img_path)
#     # dirs.sort(key=lambda x: int(x[:-4]))  # 倒着数第四位'.'为分界线，按照‘.'左边的数字从小到大排序  01.jpg
#     # dirs.sort()
#     img=
#
# def main():
#     # args = parse_args()
#     # if args.img_path is not None:
#     #     img_path=args.img_path
#     #
#     # if args.ratio_min is not None:
#     #     ratio_min=args.ratio_min
#     #
#     # if args.ratio_max is not None:
#     #     ratio_max=args.ratio_max
#     img_path='/media/n/SanDiskSSD/HardDisk/data/MNIST/Visualized/ratio_1.0/train-images/7'
#     ratio_min=0.1
#     ratio_max=1
#     auto_correlation(img_path,ratio_min,ratio_max)
#
# if __name__ == '__main__':
#     main()

import cv2
import os
import torch
import  numpy as np
import scipy.stats as stats
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
transform = transforms.Compose([
    transforms.ToTensor(),  # 0-255  转为  0-1
    transforms.Normalize((0.5,), (0.5,))  # 0-1  转为  -1-1
])
def compute_corref(img1_path,img2_path):
    img1=cv2.imread(img1_path,0)#

    img1_t=transform(img1)
    img2=cv2.imread(img2_path,0)#
    img2_t=transform(img2)
    img1_row = img1_t.squeeze_(0).reshape(-1)
    img2_row = img2_t.squeeze_(0).reshape(-1)
    # (cor,_) = stats.pearsonr(img1_row, img2_row)
    cor = torch.cosine_similarity(img1_row, img2_row, dim=0)
    # cor=np.corrcoef(img1_row.numpy(),img2_row.numpy())

    return cor

imgs_prefix='/media/n/SanDiskSSD/HardDisk/temp/new_paper/pics/20210705/1/imgs_of_7_of_dif_ratios'
img1_path='/media/n/SanDiskSSD/HardDisk/temp/new_paper/pics/20210705/1/imgs_of_7_of_dif_ratios/1.000.jpg'

# img1_path=os.path.join(imgs_prefix, img1)
img1=cv2.imread(img1_path,0)
files = os.listdir(imgs_prefix)
# 倒着数第四位 '.'为分界线，按照‘.'左边的数字从小到大排序  0.72.jpg
files.sort(key=lambda x: float(x[:-4]))  # 倒着数第四位,'.'为分界线，按照‘.'左边的数字从小到大排序  0.72.jpg

xs = []
cors = []
for file in files:
    x=float(os.path.splitext(file)[0])
    img2_path=os.path.join(imgs_prefix, file)
    cor =compute_corref(img1_path,img2_path)
    # x_scale.append(float(dir))
    cors.append(abs(cor))
    xs.append(x)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.tick_params(axis='both', which='major', labelsize=14)
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }
plt.xlabel('scale ratio', font1)
plt.ylabel('autocorrelation coefficient', font1)
plt.title('output')
plt.xticks(np.arange(0, 1.0+0.1, 0.1))
plt.yticks(np.arange(0.5,1+0.05, 0.05))
plt.xlim([0, 1])
plt.ylim([0.5, 1])
plt.grid(True)
plt.plot(xs, cors)
plt.show()











