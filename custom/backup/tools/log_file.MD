# log file
## 2021.07.22
### 标准MNIST数据集
  - 报错：‘10’ is not registered ，方法：把workers_per_gpu改为0
  - 正常训练生成.pth文件
    ```shell
    python custom/tools/train/train.py \
    custom/configs/lenet/lenet5_mnist_scaled.py \
    --work-dir ./custom/work_dirs/20210722/train/normal_cnn
    ```
  - 利用正常训练的.pth文件，进行测试
    
    my_transform.py的RandomScaling中进行[1/2.5,1.0]范围内随机缩小
    ```shell
    python ./custom/tools/test/test.py \
    custom/configs/lenet/lenet5_mnist_scaled.py \
    ./custom/work_dirs/20210722/train/normal_cnn/epoch_30.pth  \
    --metrics accuracy \
    --out custom/work_dirs/20210722/test/normal/result_0.4_1.json 
    59.82%
    ```
  - 数据增强
  
    训练测试都在RandomScaling中进行[1/2.5,1.0]范围内随机缩小
      ```shell
    python custom/tools/train/train.py \
    custom/configs/lenet/lenet5_mnist_scaled.py  \
    --work-dir ./custom/work_dirs/20210722/train/data_augmentation_1.0_2.5
    
    python custom/tools/test/test.py \
    custom/configs/lenet/lenet5_mnist_scaled.py \
    ./custom/work_dirs/20210722/train/data_augmentation_1.0_2.5/epoch_30.pth \
    --metrics accuracy \
    --out custom/work_dirs/20210722/test/data_augmentation/ratio_1.0_2.5.json
    98.92%
    ```
  - 多通道的方法
    
    利用正常训练的.pth文件，测试某一张图片，
  
    scaled_mnist.py读取图片，进行[1/2.5,1.0]范围内随机缩小.
    
    然后lenet_numtichannel.py 对该图片在[1.0,2.5]范围内，进行等间隔放大，
    
    每个通道单独进行test，取最大相应作为test结果。  
    ```shell
    python custom/tools/test/test_multichannel.py \
    custom/configs/lenet/lenet5_mnist_multichannel.py \
    ./custom/work_dirs/20210722/train/normal_cnn/epoch_30.pth  \
    --metrics accuracy \
    --out custom/work_dirs/20210722/test/my_method/r_1.0_2.5_c01.json \
    --options {'model.backbone.max_scale=2.5','model.backbone.num_channels=1','data.test.max_scale=2.5'}
    59.03%
    ```
    ```shell
    python custom/tools/test/test_multichannel.py \
    custom/configs/lenet/lenet5_mnist_multichannel.py \
    ./custom/work_dirs/20210722/train/normal_cnn/epoch_30.pth  \
    --metrics accuracy \
    --out custom/work_dirs/20210722/test/my_method/r_1.0_2.5_c02.json \
    --options {'model.backbone.max_scale=2.5','model.backbone.num_channels=02','data.test.max_scale=2.5'}
    80.77%
    ```
    ```shell
    python custom/tools/test/test_multichannel.py \
    custom/configs/lenet/lenet5_mnist_multichannel.py \
    ./custom/work_dirs/20210722/train/normal_cnn/epoch_30.pth  \
    --metrics accuracy \
    --out custom/work_dirs/20210722/test/my_method/r_1.0_2.5_c03.json \
    --options {'model.backbone.max_scale=2.5','model.backbone.num_channels=03','data.test.max_scale=2.5'}
    90.98%
    ```
    ```shell
    python custom/tools/test/test_multichannel.py \
    custom/configs/lenet/lenet5_mnist_multichannel.py \
    ./custom/work_dirs/20210722/train/normal_cnn/epoch_30.pth  \
    --metrics accuracy \
    --out custom/work_dirs/20210722/test/my_method/r_1.0_2.5_c04.json \
    --options {'model.backbone.max_scale=2.5','model.backbone.num_channels=04','data.test.max_scale=2.5'}
    93.30%
    ```
    ```shell
    python custom/tools/test/test_multichannel.py \
    custom/configs/lenet/lenet5_mnist_multichannel.py \
    ./custom/work_dirs/20210722/train/normal_cnn/epoch_30.pth  \
    --metrics accuracy \
    --out custom/work_dirs/20210722/test/my_method/r_1.0_2.5_c05.json \
    --options {'model.backbone.max_scale=2.5','model.backbone.num_channels=05','data.test.max_scale=2.5'}
    93.80%
    ```
    ```shell
    python custom/tools/test/test_multichannel.py \
    custom/configs/lenet/lenet5_mnist_multichannel.py \
    ./custom/work_dirs/20210722/train/normal_cnn/epoch_30.pth  \
    --metrics accuracy \
    --out custom/work_dirs/20210722/test/my_method/r_1.0_2.5_c10.json \
    --options {'model.backbone.max_scale=2.5','model.backbone.num_channels=10','data.test.max_scale=2.5'}
    94.36%
    ```
    ```shell
    python custom/tools/test/test_multichannel.py \
    custom/configs/lenet/lenet5_mnist_multichannel.py \
    ./custom/work_dirs/20210722/train/normal_cnn/epoch_30.pth  \
    --metrics accuracy \
    --out custom/work_dirs/20210722/test/my_method/r_1.0_2.5_c20.json \
    --options {'model.backbone.max_scale=2.5','model.backbone.num_channels=20','data.test.max_scale=2.5'}
    94.53%
    ```
    ```shell
    python custom/tools/test/test_multichannel.py \
    custom/configs/lenet/lenet5_mnist_multichannel.py \
    ./custom/work_dirs/20210722/train/normal_cnn/epoch_30.pth  \
    --metrics accuracy \
    --out custom/work_dirs/20210722/test/my_method/r_1.0_2.5_c30.json \
    --options {'model.backbone.max_scale=2.5','model.backbone.num_channels=30','data.test.max_scale=2.5'}
    94.65%
    ```
    结果表面通道数达到一定值后，top-1不随通道数增加。


##2021.07.21
不用标准MNIST数据集，改用自己生成的size=56的MNIST数据集，相当于在size=28的基础上进行padding.

计算MNIST56数据集，mean=0.025860(6.590) std=0.149996(38.249).

修改config文件中的mean和std.
  -  正常训练生成.pth文件
```shell
python custom/tools/train/train.py \
custom/configs/lenet/lenet5_mnist_56.py \
--work-dir ./custom/work_dirs/20210721/train/normal_cnn_size_56
```
  - 利用正常训练的.pth文件，测试进行[1.0,2.5]范围内随机放大
```shell
python ./custom/tools/test/test.py \
custom/configs/lenet/lenet5_mnist_56.py \
./custom/work_dirs/20210721/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210721/test/normal/result_1.0_2.5.json
67.11%
```
  - 数据增强，训练测试都进行[1.0,2.5]范围内随机放大
```shell
python custom/tools/train/train.py \
custom/configs/lenet/lenet5_mnist_56.py  \
--work-dir ./custom/work_dirs/20210721/train/data_augmentation

python custom/tools/test/test.py \
custom/configs/lenet/lenet5_mnist_56.py \
./custom/work_dirs/20210721/train/data_augmentation/epoch_30.pth \
--metrics accuracy \
--out custom/work_dirs/20210721/test/data_augmentation/ratio_1.0_2.5.json
99.20%
```
  - 多通道的方法，利用正常训练的.pth文件，测试某一张图片，进行[1.0,2.5]范围内随机放大.然后对该图片在[1.0/2.5,1.0]范围内，进行等间隔缩小，每个通道单独进行test，取最大相应作为test结果。  
  
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel_56.py \
./custom/work_dirs/20210721/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210721/test/my_method/r_1.0_2.5_c1.json 
51.06% 
```
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel_56.py \
./custom/work_dirs/20210721/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210721/test/my_method/r_1.0_2.5_c2.json 
80.81% 
```
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel_56.py \
./custom/work_dirs/20210721/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210721/test/my_method/r_1.0_2.5_c3.json 
91.60%
```
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel_56.py \
./custom/work_dirs/20210721/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210721/test/my_method/r_1.0_2.5_c4.json 
91.60%
```
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel_56.py \
./custom/work_dirs/20210721/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210721/test/my_method/r_1.0_2.5_c5.json 
94.05% 
```
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel_56.py \
./custom/work_dirs/20210721/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210721/test/my_method/r_1.0_2.5_c6.json 
94.43%
```
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel_56.py \
./custom/work_dirs/20210721/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210721/test/my_method/r_1.0_2.5_c10.json 
94.49% 
```
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel_56.py \
./custom/work_dirs/20210721/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210721/test/my_method/r_1.0_2.5_c20.json 
94.24%
```
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel_56.py \
./custom/work_dirs/20210721/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210721/test/my_method/r_1.0_2.5_c30.json 
94.39% 
```
结果表面通道数达到一定值后，top-1不随通道数增加。
##2021.07.17
训练size=96的MNIST数据集 生成.pth文件 
```shell
python custom/tools/train/train.py \
custom/configs/lenet/lenet5_mnist_56.py \
--work-dir ./custom/work_dirs/20210717/train/normal_cnn_size_56
98.99%
```
利用正常训练的数据集，测试的时候进行随机拉伸，设置

samples_per_gpu=1,workers_per_gpu=0,
```shell
python ./custom/tools/test/test.py \
custom/configs/lenet/lenet5_mnist_56.py \
./custom/work_dirs/20210717/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210717/test/normal/result_1.0_2.5.json
64.04%
```
数据增强
```shell
python custom/tools/train/train.py \
custom/configs/lenet/lenet5_mnist_56.py  \
--work-dir ./custom/work_dirs/20210717/train/data_augmentation

python custom/tools/test/test.py \
custom/configs/lenet/lenet5_mnist_56.py \
./custom/work_dirs/20210717/train/data_augmentation/epoch_30.pth \
--metrics accuracy \
--out custom/work_dirs/20210717/test/data_augmentation/ratio_1.0_2.5.json
99.05%
```
多通道的方法
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel_56.py \
./custom/work_dirs/20210717/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210717/test/my_method/r_1.0_2.5_c10.json 
91.52%
```
注意图像均值和方差,img_pad不是补零而是-1

训练测试要同样的均值方差

mean=[33.46], std=[78.87]改为：mean=[127.5], std=[127.5]

normal
```shell
python custom/tools/train/train.py \
custom/configs/lenet/lenet5_mnist_56.py \
--work-dir ./custom/work_dirs/20210720/train/normal_cnn_size_56
```
随机拉伸
```shell
python ./custom/tools/test/test.py \
custom/configs/lenet/lenet5_mnist_56.py \
./custom/work_dirs/20210720/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210720/test/normal/result_1.0_2.5.json
70.31%
```
数据增强
```shell
python custom/tools/train/train.py \
custom/configs/lenet/lenet5_mnist_56.py  \
--work-dir ./custom/work_dirs/20210720/train/data_augmentation

python custom/tools/test/test.py \
custom/configs/lenet/lenet5_mnist_56.py \
./custom/work_dirs/20210720/train/data_augmentation/epoch_30.pth \
--metrics accuracy \
--out custom/work_dirs/20210720/test/data_augmentation/ratio_1.0_2.5.json
99.02%
```
多通道的方法
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel_56.py \
./custom/work_dirs/20210720/train/normal_cnn_size_56/epoch_30.pth  \
--metrics accuracy \
--out custom/work_dirs/20210720/test/my_method/r_1.0_2.5_c10.json 
56.03%   明显有问题
```

##2021.07.16

 - 测试集前100张图片的真值标签：
 
      7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4, 9, 6, 6, 5,
      4, 0, 7, 4, 0, 1, 3, 1, 3, 4, 7, 2, 7, 1, 2, 1, 1, 7, 4, 2, 3, 5, 1, 2,
      4, 4, 6, 3, 5, 5, 6, 0, 4, 1, 9, 5, 7, 8, 9, 3, 7, 4, 6, 4, 3, 0, 7, 0,
      2, 9, 1, 7, 3, 2, 9, 7, 7, 6, 2, 7, 8, 4, 7, 3, 6, 1, 3, 6, 9, 3, 1, 4,
      1, 7, 6
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel.py \
./custom/work_dirs/20210618/train/normal/epoch_50.pth  \
--metrics accuracy \
--out custom/work_dirs/20210623/test/my_method/r_1.0_2.5_c20.json \
--options {'model.backbone.max_scale=2.5','model.backbone.num_channels=20','data.test.max_scale=2.5'}
97.10%
```
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel.py \
./custom/work_dirs/20210618/train/normal/epoch_50.pth  \
--metrics accuracy \
--out custom/work_dirs/20210623/test/my_method/r_1.0_2.0_c20.json \
--options {'model.backbone.max_scale=2.0','model.backbone.num_channels=20','data.test.max_scale=2.0'}
98.07
```
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel.py \
./custom/work_dirs/20210618/train/normal/epoch_50.pth  \
--metrics accuracy \
--out custom/work_dirs/20210623/test/my_method/r_1.0_2.5_c10.json \
--options {'model.backbone.max_scale=2.5','model.backbone.num_channels=10','data.test.max_scale=2.5'}
96.79
```   
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel.py \
./custom/work_dirs/20210618/train/normal/epoch_50.pth  \
--metrics accuracy \
--out custom/work_dirs/20210623/test/my_method/r_1.0_2.5_c50.json \
--options {'model.backbone.max_scale=2.5','model.backbone.num_channels=50','data.test.max_scale=2.5'}
96.79
```  
##2021.07.15
采用大小缩放之后的数据集
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel.py \
./custom/work_dirs/20210618/train/normal/epoch_50.pth  \
--metrics accuracy \
--out custom/work_dirs/20210623/test/my_method/r_1.0_5.0_c10.json \
--options {'model.backbone.max_scale=5.0','model.backbone.num_channels=10','data.test.max_scale=5.0'}
82.05%
```

```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel.py \
./custom/work_dirs/20210618/train/normal/epoch_50.pth  \
--metrics accuracy \
--out custom/work_dirs/20210623/test/my_method/r_1.0_5.0_c20.json \
--options {'model.backbone.max_scale=5.0','model.backbone.num_channels=20','data.test.max_scale=5.0'}
82.48%
```

```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel.py \
./custom/work_dirs/20210618/train/normal/epoch_50.pth  \
--metrics accuracy \
--out custom/work_dirs/20210623/test/my_method/r_1.0_5.0_c30.json \
--options {'model.backbone.max_scale=5.0','model.backbone.num_channels=30','data.test.max_scale=5.0'}
82.44%
```


##2021.07.05
测试SIM2MNIST


##2021.07.05
test_20210705.sh 

生成可视化的拉伸后的图片：拉伸前后目标大小不变，只是变模糊



##2021.07.03 
### 
这里的拉伸是不改变大小的拉伸，只是变模糊
- normal cnn 

    正常数据集上训练的CNN，测试时进行随机拉伸
```shell
python ./custom/tools/test/test.py \
custom/configs/lenet/lenet5_mnist_scaled.py \
./custom/work_dirs/20210618/train/normal/epoch_50.pth  \
--metrics accuracy \
--out custom/work_dirs/20210704/test/normal/result_0.1_1.json 

88.77%
```

- data augmentation
 
    数据增强（随机拉伸）的数据集上进行训练，测试时进行随机拉伸
```shell
xx.xx%
```
- my method 

    正常数据集上训练的CNN，测试时进行随机拉伸，加上自己的方法
```shell
python custom/tools/test/test_multichannel.py \
custom/configs/lenet/lenet5_mnist_multichannel.py \
./custom/work_dirs/20210618/train/normal/epoch_50.pth  \
--metrics accuracy \
--out custom/work_dirs/20210704/test/my_method/r_1.0_9.9_c20.json \
--options {'model.backbone.max_scale=9.9','model.backbone.num_channels=1','data.test.max_scale=9.9'}

73.07%
```


##2021.06.21
###random testset  size  1-5.0
- normal cnn

    65.06%
 
- data_aug

    97.83%

- my method  (channels=300)

    90.30%