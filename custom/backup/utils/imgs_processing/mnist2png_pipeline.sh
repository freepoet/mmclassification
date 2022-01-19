#!/bin/bash
cd ../../


for i in $(seq 1.0 0.1 2)
do
  python  ./custom/utils/imgs_processing/mnist2png_with_scaled_height.py--mnist_path '/media/n/SanDiskSSD/HardDisk/data/' --png_path_prefix '/media/n/SanDiskSSD/HardDisk/data/MNIST/Visualized' --scale_ratio $i
done
