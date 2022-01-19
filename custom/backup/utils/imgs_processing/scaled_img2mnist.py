import os
from PIL import Image
from array import *
from random import shuffle
import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='scaled_img2mnist')
    parser.add_argument('--img_path', help='Imgs')
    args = parser.parse_args()
    return args

def png2mnist(img_path):

    # Load from and save to
    Names = [[img_path + 'train-images', 'train'], [img_path + 'test-images', 't10k']]

    for name in Names:

        data_image = array('B')
        data_label = array('B')

        FileList = []
        for dirname in os.listdir(name[0])[1:]:  # [1:] Excludes .DS_Store from Mac OS
            path = os.path.join(name[0], dirname)
            for filename in os.listdir(path):
                if filename.endswith(".png"):
                    FileList.append(os.path.join(name[0], dirname, filename))

        shuffle(FileList)  # Usefull for further segmenting the validation set

        for filename in FileList:

            label = int(filename.split('/')[10])

            Im = Image.open(filename)

            pixel = Im.load()

            width, height = Im.size

            for x in range(0, width):
                for y in range(0, height):
                    # data_image.append(pixel[y,x])
                    data_image.append(pixel[x, y])

            data_label.append(label)  # labels start (one unsigned byte each)

        hexval = "{0:#0{1}x}".format(len(FileList), 6)  # number of files in HEX

        # header for label array

        header = array('B')
        header.extend([0, 0, 8, 1, 0, 0])
        header.append(int('0x' + hexval[2:][:2], 16))
        header.append(int('0x' + hexval[2:][2:], 16))

        data_label = header + data_label

        # additional header for images array

        if max([width, height]) <= 256:
            header.extend([0, 0, 0, width, 0, 0, 0, height])
        else:
            raise ValueError('Image exceeds maximum size: 256x256 pixels');

        header[3] = 3  # Changing MSB for image data (0x00000803)

        data_image = header + data_image

        out_path=img_path + 'MNIST/raw/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        output_file = open(out_path+ name[1] + '-images-idx3-ubyte', 'wb')
        data_image.tofile(output_file)
        output_file.close()

        output_file = open(out_path + name[1] + '-labels-idx1-ubyte', 'wb')
        data_label.tofile(output_file)
        output_file.close()

    # gzip resulting files

    for name in Names:
        os.system('gzip ' + out_path + name[1] + '-images-idx3-ubyte')  # accelerate
        os.system('gzip ' + out_path + name[1] + '-labels-idx1-ubyte')



def main():
    args = parse_args()
    if args.img_path is not None:
        img_path=args.img_path
    # img_path = '/media/n/SanDiskSSD/HardDisk/data/MNIST/Visualized/height_2.0/'
    png2mnist(img_path)
    print('convert successfully!\n')

if __name__ == '__main__':
    main()

# python custom/my_own/scaled_img2mnist.py --img_path /media/n/SanDiskSSD/HardDisk/data/MNIST/Visualized/ratio_1.0/