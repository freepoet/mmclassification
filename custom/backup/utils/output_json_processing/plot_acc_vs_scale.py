# json: 用于字符串和python数据类型间进行转换
# pickle： 用于python特有的类型和python的数据类型间进行转换
import json
import os
import matplotlib.pyplot as plt
import numpy as  np
def obtain_max_acc(json_file_path):
    '''
    
    
    Returns:

    '''
    json_data=[]

    with open(json_file_path) as f:
        try:
            for line in f:
                json_data.append(json.loads(line))
        except:
            #shang shu yi chang kuai de chu li fang fa
            print("json format is not corrrect")
            exit(1)


    accuracy_1=[]
    key_name="accuracy_top-1"
    for data in json_data:
        # if data.has_key(key_name): 字典的has_key方法只能在Python2中使用，在Python3中已经移除。
        if  key_name in data.keys():
            accuracy_1.append(data[key_name])
    accuracy_1_max=max(accuracy_1)
    return accuracy_1_max


if __name__ == '__main__':
    root_path='/home/n/Github/mmclassification0110/mmclassification/custom/work_dirs/mnist/lenet5_mnist_scaled/ratio_h'
    dirs = os.listdir(root_path)
    # dirs.sort(key=lambda x: int(x[:-4]))  # 倒着数第四位'.'为分界线，按照‘.'左边的数字从小到大排序  01.jpg
    dirs.sort()
    x_scale=[]
    y_max_acc=[]
    for dir in dirs:
        files = os.listdir(os.path.join(root_path,dir))
        for file in files:
            if file.endswith('.json'):
                json_file_path=os.path.join(root_path,dir,file)
                max_acc=obtain_max_acc(json_file_path)
                x_scale.append(float(dir))
                y_max_acc.append(max_acc)
    # 刻度朝内
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.tick_params(axis='both', which='major', labelsize=14)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    plt.xlabel('ratio h', font1)
    plt.ylabel('accuracy_top1/%', font1)

    plt.xticks(np.arange(0.1, 1.0+0.1, 0.1))
    plt.yticks(np.arange(0,100+10, 10))
    plt.xlim([0.1, 1])
    plt.ylim([0, 100])
    plt.grid(True)

    plt.plot(x_scale,y_max_acc)
    plt.show()