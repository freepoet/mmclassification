# !/bin/bash
cd ../../../

for ratio in $(seq 0 0.01 1)
do
  python custom/tools/test/test_w_h.py custom/configs/lenet/lenet5_mnist_multichannel.py ./custom/work_dirs/20210618/train/normal/epoch_50.pth   --ratio-w $ratio --ratio-h $ratio --metrics accuracy --out custom/work_dirs/temp/mnist/ratio_$ratio.json
done
