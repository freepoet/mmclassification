#!/bin/bash
cd ../../../

for ratio_h in $(seq 1.0 -0.002 0.1)
do
#  echo $ratio_h
  python custom/tools/train/train.py custom/configs/lenet/lenet5_mnist_scaled.py --ratio-h $ratio_h --work-dir ./custom/work_dirs/mnist/lenet5_mnist_scaled/ratio_h/$ratio_h --gpu-ids 5
done
#  python custom/tools/train/train.py custom/configs/lenet/lenet5_mnist_scaled.py --work-dir ./custom/work_dirs/temp
# python custom/tools/train/train.py custom/configs/lenet/lenet5_mnist_scaled.py --work-dir ./custom/work_dirs/temp

#  python custom/tools/train_with_dif_ratio_w_h/train_with_dif_ratio_w_h.py custom/configs/lenet/lenet5_mnist_scaled.py  --work-dir ./custom/work_dirs/20210615/train/data_au


### 0.1-10
## normal cnn
# normal test
# python custom/tools/train/train.py custom/configs/lenet/lenet5_mnist_scaled.py  --work-dir ./custom/work_dirs/20210615/train/normal
# randomscaling test
# python custom/tools/train/train.py custom/configs/lenet/lenet5_mnist_scaled.py  --work-dir ./custom/work_dirs/20210615/train/normal_random_scaling_test
#
#
#
## normal cnn with data aug
# normal test
# python custom/tools/train/train.py custom/configs/lenet/lenet5_mnist_scaled.py  --work-dir ./custom/work_dirs/20210615/train/da_train_normal_test
# randomscaling test
# python custom/tools/train/train.py custom/configs/lenet/lenet5_mnist_scaled.py  --work-dir ./custom/work_dirs/20210615/train/da_train_rd_test

### 0.1-1
## normal cnn
# normal test
# python custom/tools/train/train.py custom/configs/lenet/lenet5_mnist_scaled.py  --work-dir ./custom/work_dirs/20210618/train/normal
# randomscaling test
# python custom/tools/train/train.py custom/configs/lenet/lenet5_mnist_scaled.py  --work-dir ./custom/work_dirs/20210618/train/normal_random_scaling_test
#
#
#
## normal cnn with data aug
# normal test
# python custom/tools/train/train.py custom/configs/lenet/lenet5_mnist_scaled.py  --work-dir ./custom/work_dirs/20210618/train/da_train_normal_test
# randomscaling test
# python custom/tools/train/train.py custom/configs/lenet/lenet5_mnist_scaled.py  --work-dir ./custom/work_dirs/20210618/train/da_train_rd_test
