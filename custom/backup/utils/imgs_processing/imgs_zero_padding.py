import os
import numpy as np
from PIL import Image
import argparse
import ipdb
def parse_args():

    parser = argparse.ArgumentParser(description='imgs with zero padding')
    parser.add_argument('--ratio', help=' ratio')

    args = parser.parse_args()
    return args

def img_crop(pil_file,single_size):
    w, h = pil_file.size
    fixed_size = single_size  # 输出正方形图片的尺寸
    array_file=pil_file.crop((w/2-fixed_size/2,h/2-fixed_size/2,w/2+fixed_size/2,h/2+fixed_size/2))
    # ipdb.set_trace()

    # output_file = Image.fromarray(array_file)
    return np.array(array_file)

def img_pad(pil_file,single_size):
    w, h = pil_file.size
    fixed_size = single_size  # 输出正方形图片的尺寸

    if h >= w:
        factor = h / float(fixed_size)
        new_w = int(w / factor)
        if new_w % 2 != 0:
            new_w -= 1
        pil_file = pil_file.resize((new_w, fixed_size))
        pad_w = int((fixed_size - new_w) / 2)
        array_file = np.array(pil_file)
        array_file = np.pad(array_file, ((0, 0), (pad_w, pad_w)), 'constant')
    else:
        factor = w / float(fixed_size)
        new_h = int(h / factor)
        if new_h % 2 != 0:
            new_h -= 1
        pil_file = pil_file.resize((fixed_size, new_h))
        pad_h = int((fixed_size - new_h) / 2)
        array_file = np.array(pil_file)
        array_file = np.pad(array_file, ((pad_h, pad_h), (0, 0)), 'constant')

    return array_file

def my_img_pad(pil_file,single_size):
    w, h = pil_file.size
    if w % 2 != 0:
        w=w+1
        h=w
        pil_file = pil_file.resize((w, h))
    fixed_size = single_size  # 输出正方形图片的尺寸
    padding_w=int((fixed_size-w)/2)
    padding_h=padding_w
    array_file = np.array(pil_file)
    array_file = np.pad(array_file, ((padding_w, padding_h), (padding_w, padding_h)), 'constant',constant_values = (-1,-1))

    return array_file

# if __name__ == "__main__":
#
#     args = parse_args()
#     if args.ratio is not None:
#         ratio=args.ratio
#
#     dir_prefix = '/media/n/SanDiskSSD/HardDisk/data/MNIST/Visualized/'+ ratio+'/' # 图片所在文件夹
#     # dir_prefix = '/media/n/SanDiskSSD/HardDisk/data/MNIST/Visualized/ratio_2.0/'  # 图片所在文件夹
#     dir_output_prefix=dir_prefix+'padding/'  # 输出结果文件夹
#
#     dir_suffix_first = ['train-images/', 'test-images/']
#     dir_suffix_last = ['0', '1','2','3','4','5','6','7','8','9']
#     for dir1 in dir_suffix_first:
#         for dir2 in dir_suffix_last:
#             dir_image=dir_prefix+dir1+dir2
#             i = 0
#             list_image = os.listdir(dir_image)
#             for file in list_image:
#                 path_image = os.path.join(dir_image, file)
#                 dir_output=dir_output_prefix+dir1+dir2
#                 if not os.path.exists(dir_output):
#                     os.makedirs(dir_output)
#                 path_output = os.path.join(dir_output, file)
#
#                 pil_image = Image.open(path_image).convert('L')
#                 output_image = img_pad(pil_image)
#                 output_image.save(path_output)
#                 i += 1
#                 # print('The num of processed images:', i)
#     print('Processing'+ratio+'finished')