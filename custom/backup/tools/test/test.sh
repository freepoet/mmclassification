# !/bin/bash
cd ../../../
#
#for ratio in $(seq 0 0.002 1)
#do
#  python custom/tools/test/test_w_h.py custom/configs/lenet/lenet5_mnist_multichannel.py ./custom/work_dirs/20210618/train/normal/epoch_50.pth   --ratio-w $ratio --ratio-h $ratio --metrics accuracy --out custom/work_dirs/temp/mnist/ratio_$ratio.json
#done
#for ratio in $(seq 1 1 12)
#do
##  echo $ratio
#  python custom/tools/test/test_w_h.py custom/configs/lenet/lenet5_mnist_multichannel.py ./custom/work_dirs/20210618/train/normal/epoch_50.pth   --ratio-w $ratio --ratio-h $ratio --metrics accuracy --out custom/work_dirs/temp/mnist/ratio_$ratio.json
#done


#python custom/tools/test_with_dif_ratio_w_h/test_plotting_cm.py custom/configs/lenet/lenet5_mnist_scaled.py  custom/work_dirs/20210607/mnist/lenet5_mnist_scaled/w_h_1.0/epoch_10.pth  --ratio-w $ratio --ratio-h $ratio --metrics accuracy --out custom/work_dirs/20210609/mnist/test_id_w_h/ratio_$ratio.json
## random testset  size  0.1-10
# normal cnn
#python custom/tools/test/test.py custom/configs/lenet/lenet5_mnist_scaled.py ./custom/work_dirs/20210615/train/normal/epoch_50.pth  --metrics accuracy --out custom/work_dirs/20210615/test/normal/ratio_$ratio.json
#  30.45%
#python custom/tools/test/test.py custom/configs/lenet/lenet5_mnist_scaled.py ./custom/work_dirs/20210615/train/normal_random_scaling_test/epoch_50.pth  --metrics accuracy --out custom/work_dirs/20210615/test/normal_random_scaling_test/ratio_$ratio.json
#  26.30%
#
#
# data_aug cnn  xxx%   both are random_scaling in test
#python custom/tools/test/test.py custom/configs/lenet/lenet5_mnist_scaled.py ./custom/work_dirs/20210615/train/da_train_normal_test/epoch_50.pth --metrics accuracy --out custom/work_dirs/20210615/test/da_train_normal_test/ratio_$ratio.json
#  83.51%
# python custom/tools/test/test.py custom/configs/lenet/lenet5_mnist_scaled.py ./custom/work_dirs/20210615/train/da_train_rd_test/epoch_50.pth --metrics accuracy --out custom/work_dirs/20210615/test/da_train_rd_test/ratio_$ratio.json
#  83.20%

## trian:normal_cnn      test:random_scaling + my_method
# python custom/tools/test/test_multichannel.py custom/configs/lenet/lenet5_mnist_multichannel.py ./custom/work_dirs/20210615/train/normal/epoch_50.pth  --metrics accuracy --out custom/work_dirs/20210617/test/my_method/ratio_$ratio.json
#  23.69 %


## random testset  size  0.1-1
# normal cnn
#python custom/tools/test/test.py custom/configs/lenet/lenet5_mnist_scaled.py ./custom/work_dirs/20210618/train/normal/epoch_50.pth  --metrics accuracy --out custom/work_dirs/20210618/test/normal/ratio_$ratio.json
#  88.97%
#python custom/tools/test/test.py custom/configs/lenet/lenet5_mnist_scaled.py ./custom/work_dirs/20210618/train/normal_random_scaling_test/epoch_50.pth  --metrics accuracy --out custom/work_dirs/20210618/test/normal_random_scaling_test/ratio_$ratio.json
#  88.81%
#
# data_aug cnn  xxx%   both are random_scaling in test
#python custom/tools/test/test.py custom/configs/lenet/lenet5_mnist_scaled.py ./custom/work_dirs/20210618/train/da_train_normal_test/epoch_50.pth --metrics accuracy --out custom/work_dirs/20210618/test/da_train_normal_test/ratio_$ratio.json
#  94.46%
# python custom/tools/test/test.py custom/configs/lenet/lenet5_mnist_scaled.py ./custom/work_dirs/20210618/train/da_train_rd_test/epoch_50.pth --metrics accuracy --out custom/work_dirs/20210618/test/da_train_rd_test/ratio_$ratio.json
#  94.77%


## trian:normal_cnn      test:random_scaling + my_method

#python custom/tools/test/test_multichannel.py custom/configs/lenet/lenet5_mnist_multichannel.py ./custom/work_dirs/20210618/train/normal/epoch_50.pth  --metrics accuracy --out custom/work_dirs/20210618/test/my_method/ratio_$ratio.json
# 1-9.9 90  channels 79.33 %
# 1-9.9 360 channels 79.58 %



## random testset  size  1-5.0
# normal cnn
# 65.06%
# data_aug
# 97.83%
# my method  300c
# 90.30%

#num_channels=100
#for max_scale in $(seq 1.0 0.1 9.9)
#do
#  python custom/tools/test/test_multichannel.py custom/configs/lenet/lenet5_mnist_multichannel.py ./custom/work_dirs/20210618/train/normal/epoch_50.pth  --metrics accuracy --out custom/work_dirs/20210621/test/100_chs_dif_scale/max_scale_$max_scale.json   --options {'model.backbone.max_scale='$max_scale,'model.backbone.num_channels='$num_channels,'data.test.max_scale='$max_scale}
#done



