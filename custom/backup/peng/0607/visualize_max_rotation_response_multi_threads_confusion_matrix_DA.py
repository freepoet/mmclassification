# Src  : max_rotation_response.py, by Peng Zhang
# Attr : 2020/12/11 下午5:06, pengloveai@163.com
# Func :
import sys
sys.path.append("../../")
# File IO
import os
import shutil
# Image Processing
# import cv2
import numpy as np
# Model
import torch as t
import torch.nn as nn
from peer_models import SingleBranchBN as TIPNet
from analytic_utils import plot_confusion_matrix, confusion_matrix
# import genotypes
# from model import NetworkCIFAR as Network  # custom NAS net arch
# Optimizer
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# Dataset
# from torchvision.datasets import MNIST
from rotated_mnist import RandomRotatedMNIST_int as RotatedMNIST
# Progress Bar
from tqdm import tqdm
import time

# Set GPU device ID
t.cuda.set_device(0)
N_angle = 3600
N_angle = 360
# Define Dataset
mnist_path = "../mnist"
# mnist_path = "/home/p/Documents/experiment/rotation_pattern_recognition/PreRot/max_rotation_response_darts0106/mnist"
n_classes = 10
mean, std = 0.1307, 0.3081
train_dataset = RotatedMNIST(mnist_path, train=True, download=False,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((mean,), (std,))])
                      )
test_dataset = RotatedMNIST(mnist_path, train=False, download=False,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((mean,), (std,))]),
                     init_range=360,
                     N=N_angle
                     )
# ====== Check data ======
# a = test_dataset[0]
# print(type(a[0]))
# print(a[0].dtype)
# 创建data loaders, Create data loaders
batch_size = 2
CUDA = t.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# TODO: 1. Define Model, choose proper genotype & Net shape
# genotype = eval("genotypes.%s" % 'PCDARTS')  # PCDARTS
# model_custom_cnn = Network(16, 10, 6, False, genotype)  # 16,10,8,False
# # 需要动态添加drop_path_prob属性
# model_custom_cnn.drop_path_prob = 0.3

# genotype = eval("genotypes.%s" % args.arch)  # PCDARTS
# model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype) # 16,10,8,False
NUMBER_OF_FILTERS = 40
NUMBER_OF_FC_FEATURES = 5120
model_custom_cnn = TIPNet(NUMBER_OF_FILTERS, NUMBER_OF_FC_FEATURES)


if CUDA:
    model_custom_cnn.cuda()

# Define (optimizer and loss function.)
optimizer = optim.Adam(model_custom_cnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()


def train_model(model, dataloader, loss_func, optimizer, CUDA=True, epochs=10):
    # Switch to train mode (for things like Batch norm and dropout).
    model.train()
    loss_history = []
    for epoch in range(epochs):
        for i_batch, (x_batch, y_batch) in enumerate(dataloader):
            # Compute output and loss.
            if CUDA:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            output = model(x_batch)
            loss = loss_func(output, y_batch)

            # Zero our gradients/backprop and perform and SGD step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(loss)
        loss_history.append(loss)

    return loss_history


def test_model(model, dataloader, cuda=True):
    model.eval()
    num_correct = 0
    with t.no_grad():
        for i_batch, (x_batch, y_batch) in enumerate(dataloader):
            if CUDA:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            output, _ = model(x_batch)
            # print(output.shape)
            _, output = t.max(output, dim=1)
            #             print(type(output),output.shape, output.dtype)
            #             print(type(y_batch),y_batch.shape, output.dtype)

            num_correct += t.sum(output == y_batch)

    return num_correct


def test_MRRM(model_custom_cnn, val_loader):
    # 测试输出,计算旋转图像输入的输出分数
    def check_output(images):
        with t.no_grad():
            model_custom_cnn.eval()
            images = images.cuda()
            scores = model_custom_cnn(images)
        return scores.data.cpu().numpy()  # , labels_target

    def eval_model(model, dataloader, cuda=True):
        model.eval()

        num_correct = 0

        # 记录分类准确度
        pred_label_global = 0
        pred_label_half = 0
        pred_label_quarter = 0
        # pred_label_eighth = 0

        # # 以0度为中心角度进行分类、估计角度，坐标轴设计
        x = np.array(test_dataset.ang)  # 0,1,2,...,359.0

        # y轴坐标取值
        y = np.linspace(0, 9, 10)  # 0,1,2,...,9

        # 角度估计准确度
        # 正确估计区间
        delta_ang = test_dataset.delta_angle
        ang_right = x[-1]
        # N_ang = test_dataset.N
        Ang_left_5, Ang_right_5 = 5, ang_right-5
        Ang_left_10, Ang_right_10 = 10, ang_right-10
        Ang_left_15, Ang_right_15 = 15, ang_right-15
        # Ang_left, Ang_right = -26, 25
        # 正确估计计数
        pred_ang_global_5 = 0
        pred_ang_half_5 = 0
        pred_ang_quarter_5 = 0
        pred_ang_eighth_5 = 0
        pred_ang_global_10 = 0
        pred_ang_half_10 = 0
        pred_ang_quarter_10 = 0
        pred_ang_eighth_10 = 0
        pred_ang_global_15 = 0
        pred_ang_half_15 = 0
        pred_ang_quarter_15 = 0
        pred_ang_eighth_15 = 0
        cm_g = t.zeros(10,10)
        cm_h = t.zeros(10,10)
        cm_q = t.zeros(10,10)
        with t.no_grad():
            # tqdm(total=dataloader.__len__())
            dl = dataloader.__len__()
            for i_batch, (x_batch, y_batch) in tqdm(enumerate(dataloader)):
                # print(x_batch.shape)
                tqdm.write("{}th batch of {} batches".format(i_batch, dl))
                y_batch, random_init = y_batch
                list_label_global = []
                list_label_half = []
                list_label_quarter = []
                list_label_eighth = []
                # 逐个图像计算旋转得分
                # print(x_batch.shape)
                time_start_tmp = time.time()
                img_rotated_arr_tensor2 = t.cat((x_batch[0],x_batch[1]))
                SCORES_all = check_output(img_rotated_arr_tensor2)
                SCORES_list = (SCORES_all[:N_angle,:],SCORES_all[N_angle:,:])

                for SCORES in SCORES_list:


                    # print ("scores的尺寸是", SCORES.shape)  # = 3600 x 10

                    SCORES_sampled = SCORES
                    SCORES_min = SCORES.min()
                    SCORES_sampled_shape = SCORES_sampled.shape

                    # global
                    index1 = np.unravel_index(SCORES_sampled.argmax(), SCORES_sampled_shape)
                    ang_predict1 = x[index1[0]]
                    if ang_predict1 < Ang_left_15 or ang_predict1 > Ang_right_15:
                        pred_ang_global_15 += 1
                        if ang_predict1 < Ang_left_10 or ang_predict1 > Ang_right_10:
                            pred_ang_global_10 += 1
                            if ang_predict1 < Ang_left_5 or ang_predict1 > Ang_right_5:
                                pred_ang_global_5 += 1
                    # label
                    list_label_global.append(int(y[index1[1]]))

                    # half
                    N_left_h = 900
                    N_right_h = 2700
                    SCORES_sampled[N_left_h:N_right_h,:] = SCORES_min
                    index2 = np.unravel_index(SCORES_sampled.argmax(), SCORES_sampled_shape)
                    ang_predict2 = x[index2[0]]
                    if ang_predict2 < Ang_left_15 or ang_predict2 > Ang_right_15:
                        pred_ang_half_15 += 1
                        if ang_predict2 < Ang_left_10 or ang_predict2 > Ang_right_10:
                            pred_ang_half_10 += 1
                            if ang_predict2 < Ang_left_5 or ang_predict2 > Ang_right_5:
                                pred_ang_half_5 += 1

                    list_label_half.append(int(y[index2[1]]))

                    # quarter
                    N_left_q = 450
                    N_right_q = 3150
                    SCORES_sampled[N_left_q:N_left_h, :] = SCORES_min
                    SCORES_sampled[N_right_h:N_right_q,:] = SCORES_min
                    index3 = np.unravel_index(SCORES_sampled.argmax(), SCORES_sampled_shape)
                    ang_predict3 = x[index3[0]]
                    if ang_predict3 < Ang_left_15 or ang_predict3 > Ang_right_15:
                        pred_ang_quarter_15 += 1
                        if ang_predict3 < Ang_left_10 or ang_predict3 > Ang_right_10:
                            pred_ang_quarter_10 += 1
                            if ang_predict3 < Ang_left_5 or ang_predict3 > Ang_right_5:
                                pred_ang_quarter_5 += 1

                    list_label_quarter.append(int(y[index3[1]]))

                    # # eighth
                    # N_left = 158
                    # N_right = 201
                    # SCORES_sampled = SCORES[N_left:N_right, :]
                    # index4 = np.unravel_index(SCORES_sampled.argmax(), SCORES_sampled.shape)
                    #
                    # if Ang_left < x[N_left:N_right][index4[0]] < Ang_right:
                    #     pred_ang_eighth_5 += 1
                    #
                    # list_label_eighth.append(int(y[index4[1]]))

                time_end_tmp = time.time()
                print(i_batch, "th batch inference time:", (time_end_tmp - time_start_tmp), "s")
                # 当前batch旋转多通道分类得分list---->np.array---->torch.tensor
                list_label_global = t.from_numpy(np.array(list_label_global))
                list_label_half = t.from_numpy(np.array(list_label_half))
                list_label_quarter = t.from_numpy(np.array(list_label_quarter))
                # list_label_eighth = t.from_numpy(np.array(list_label_eighth))
                cm_g = confusion_matrix(t.tensor(list_label_global), labels=y_batch, conf_matrix=cm_g)
                # cm_h = confusion_matrix(t.tensor(list_label_global), labels=y_batch, conf_matrix=cm_h)
                # cm_q = confusion_matrix(t.tensor(list_label_global), labels=y_batch, conf_matrix=cm_q)

                pred_label_global += t.sum(list_label_global == y_batch)
                pred_label_half += t.sum(list_label_half == y_batch)
                pred_label_quarter += t.sum(list_label_quarter == y_batch)
                # pred_label_eighth += t.sum(list_label_eighth == y_batch)

        # plot_confusion_matrix(cm_g.numpy(), normalize=False, title='Confusion Matrix of TIPNet(BN)+MRRM(rot360)')
        plot_confusion_matrix(cm_g.numpy(), normalize=False, title='TIPNet(BN)+PROAI (rot0-rot360)')
        # plot_confusion_matrix(cm_h.numpy(), normalize=False, title='Confusion Matrix of TIPNet(BN)+MRRM(rot180)')
        # t.save(cm_g,"./confusion_matrix360/cm_g.pt")
        # t.save(cm_h,"./confusion_matrix360/cm_h.pt")
        # t.save(cm_q,"./confusion_matrix360/cm_q.pt")
        denominator = len(val_loader) * batch_size
        return float(pred_label_global) / denominator, float(pred_label_half) / denominator, float(
            pred_label_quarter) / denominator, \
               float(pred_ang_global_5) / denominator, float(pred_ang_half_5) / denominator, float(
            pred_ang_quarter_5) / denominator, \
               float(pred_ang_global_10) / denominator, float(pred_ang_half_10) / denominator, float(
            pred_ang_quarter_10) / denominator, \
               float(pred_ang_global_15) / denominator, float(pred_ang_half_15) / denominator, float(
            pred_ang_quarter_15) / denominator, \




    p_global, p_half, p_quarter,\
    a_global_5, a_half_5, a_quarter_5,\
    a_global_10, a_half_10, a_quarter_10, \
    a_global_15, a_half_15, a_quarter_15 = eval_model(model_custom_cnn, val_loader)
    print(p_global, p_half, p_quarter,'\n',
          a_global_5, a_half_5, a_quarter_5,'\n',
          a_global_10, a_half_10, a_quarter_10,'\n',
          a_global_15, a_half_15, a_quarter_15)


def save_checkpoint(state, save_path, weight_name='checkpoint.pth.tar', best_weight_name='model_best.pth.tar', is_best=False):
    filename = os.path.join(save_path, weight_name)
    t.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_path, best_weight_name)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    load_state_sign = True
    if load_state_sign:
        state_dict_path = "./model_best.pth.tar"
        checkpoint = t.load(state_dict_path, map_location="cuda:0")
        print("epoch: {}, maximum valiadtion accuracy: {}".format(checkpoint['epoch'], checkpoint['best_valid_acc']))
        model_custom_cnn.load_state_dict(checkpoint['state_dict'])
        print("success load")
        # 测试结果
        # num_correct = test_model(model_custom_cnn, val_loader)
        # accuracy = float(num_correct) / len(val_loader) / batch_size
        # print(accuracy)
        time_start = time.time()
        test_MRRM(model_custom_cnn, val_loader)
        time_end = time.time()
        print("overall time:", (time_end-time_start)/60, "min")
    else:
        # Train our model.
        train_model(model_custom_cnn, train_loader, loss_func, optimizer, CUDA, epochs=30)
        # 模型状态保存路径设置
        savepath = "./checkpoints/diy_net"
        save_checkpoint({
            'epoch': 15,
            'state_dict': model_custom_cnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, savepath, weight_name="model_custom_cnn.pth.tar")
        # 测试结果
        num_correct = test_model(model_custom_cnn, val_loader)
        accuracy = float(num_correct) / len(val_loader) / batch_size
        print(accuracy)

