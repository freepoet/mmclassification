# Src  : rotated_mnist.py.py, by Peng Zhang
# Attr : 1/14/21 10:10 PM, pengloveai@163.com
# Func :
import torch
from torchvision.datasets import MNIST
import numpy as np
from PIL import Image
import cv2
import random
import ipdb
import os
class ScaledMNIST(MNIST):
    """
    Custom rotated MNIST
    """
    # def __init__(self, root, train=True, transform=None, target_transform=None, download=False, sign=()):
    #     super(ScaledMNIST, self).__init__(root, train, transform, target_transform, download)

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(ScaledMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        # self.Phi = Phi
        # self.N = N
        # delta_angle = Phi/N  # 0.1
        # # 计算旋转矩阵
        # img_height, img_width = 28, 28
        # img_center = (img_width // 2, img_height // 2)
        # # N = 0.1
        # # ang = list(range(0, 360, N))
        # ang = np.linspace(0, Phi-delta_angle, N)
        # ang_len = len(ang)
        # rotation_matrix = np.empty((ang_len, 2, 3))
        # for i in range(0, ang_len, 1):
        #     #     print(i)
        #     rotation_matrix[i, :, :] = cv2.getRotationMatrix2D(img_center, ang[i], 1)
        # self.rotation_matrix = rotation_matrix
        # self.ang = ang
        # self.delta_angle = delta_angle
        # self.train = train


    def __getitem__(self, index):   # scale transformation
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))


    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')
    #
    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index
    #
    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #
    #     # 图像旋转插值函数
    #     def rot3D(img3D, rotation_matrix, shape=(28, 28), pad=-0.42421296):
    #         """
    #         输入img是个二维tensor， shape=torch.Size(W,H)
    #         rotation_matrix为计算好的旋转矩阵
    #         # 输入angle是角度，作为rotation_matrix的索引
    #         shape为旋转插值后的输出
    #         pad为边界值 # pad = -0.42421296
    #         """
    #         # print(img3D.shape)
    #         img = img3D[0, :, :]
    #         NN = len(rotation_matrix)
    #         img_rotated_arr = np.empty((NN, 1, shape[0], shape[1]), dtype=np.float32)
    #         img = img.numpy()
    #         for i in range(NN):
    #             img_rotated_arr[i, 0, :, :] = cv2.warpAffine(img, rotation_matrix[i, :, :], shape, borderValue=pad)
    #         return img_rotated_arr
    #
    #     img, target = self.data[index], int(self.targets[index])
    #
    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img.numpy(), mode='L')
    #
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #
    #     if self.transform is not None:
    #         img = self.transform(img)
    #         if self.train:
    #             return img, target
    #         else:
    #             # Test dataset, rotate the image tensor N times
    #             # 旋转当前测试图像
    #             img_rotated_arr = rot3D(img, self.rotation_matrix, pad=0)
    #             # ======== Check data ======
    #             # print(img_rotated_arr.shape)
    #             # img_rotated_arr.shape = 360 x C x H x W
    #             # print(type(img_rotated_arr))
    #             # print(img_rotated_arr.dtype)
    #             return torch.from_numpy(img_rotated_arr), target
    #             # return img_rotated_arr, target
    #     return img, target