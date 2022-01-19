# Src  : analytic_utils.py, by Peng Zhang
# Attr : 1/29/21 4:44 AM, pengloveai@163.com
# Func :
# 绘制混淆矩阵
import matplotlib.pyplot as plt
import itertools
import numpy as np
from matplotlib.font_manager import FontProperties
font_times = FontProperties(fname=r"/usr/share/fonts/zh/times.ttf",size=14)
font_song = FontProperties(fname=r"/usr/share/fonts/zh/STSONG.TTF",size=14)

# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_confusion_matrix(cm, classes=["0","1","2","3","4","5","6","7","8","9",], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    # plt.figure(figsize=(6, 8))
    # plt.figure(figsize=(6, 6), dpi=600)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, FontProperties=font_times)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90,FontProperties=font_times)
    plt.yticks(tick_marks, classes,FontProperties=font_times)

    # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    # plt.tight_layout()
    plt.ylabel('True label', FontProperties=font_times)
    plt.xlabel('Predicted label',FontProperties=font_times)
    plt.savefig('/home/p/Desktop/pic/' + title + '.png', dpi=600)
    plt.savefig('/home/p/Desktop/pic/' + title + '.svg')
    plt.show()
