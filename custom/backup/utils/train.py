import torch
# torch.cuda.set_device(0)
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage()   #0-1 转 0-255
import cv2
import numpy as np
# 读取MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 0-255  转为  0-1
    transforms.Normalize((0.5,), (0.5,))  # 0-1  转为  -1-1
])
trainset = torchvision.datasets.MNIST(
    root='E:\HardDisk\data',
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
    root='E:\HardDisk\data',
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
    img_plot=(img+1)/2*255
    mat=np.ones((28,28))*255
    cv2.imwrite('./test/test.jpg',mat-img_plot)
    a=1

# class LeNet5(nn.Module):
#
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2),
#
#             nn.Conv2d(6, 16, kernel_size=5, stride=1),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2),
#
#             nn.Conv2d(16, 120, kernel_size=5, stride=1),
#             nn.Tanh()                   #  linear 16*5*5--->120  equals to conv
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(120, 84),
#             nn.Tanh(),
#             nn.Linear(84, 10),
#             )
#
#     def forward(self, x):
#
#         x = self.features(x)
#         x = self.classifier(x.squeeze())
#
#         return x
#
# net=LeNet5()
# # print(net)
#
# from torch import optim
#
# # 定义损失函数
# criterion = nn.CrossEntropyLoss()
# # 定义优化器
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# # 定义网络
# # 流程：输入数据——>梯度清零——>正向传播/反向传播——>参数更新
# for epoch in range(2):
#     print('Training start')
#     running_loss = 0.0
#     # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组、字符串、迭代器或其他支持迭代对象)组合为一个索引序列，
#     # 同时列出数据和数据下标
#     for i, data in enumerate(trainloader, 0):  # 0代表起始位置
#         # 输入数据
#         inputs, labels = data
#         # inputs,labels=Variable(inputs),Variable(labels)
#
#         # 梯度清零
#         optimizer.zero_grad()
#         # 前向 反向传播
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         # 参数更新
#         optimizer.step()
#
#         # 打印log信息
#         running_loss += loss
#         if i % 2000 == 1999:  # 2000次iter(一次iter，一个batch，4张images）打印一次
#             print('[%d,%5d] loss:%.3f' \
#                   % (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
# print('Training finished')
#
# # correct = 0
# # total = 0
# # for data in testloader:
# #     images, labels = data
# #     outputs = net(images)
# #     _, predicted = torch.max(outputs.data, 1)
# #     total += labels.size(0)
# #     correct += (predicted == labels).sum()
# # print('%f%%' % (correct * 100 / total))