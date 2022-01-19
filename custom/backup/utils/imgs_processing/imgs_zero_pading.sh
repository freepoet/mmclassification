#!/bin/bash
cd ../../../
for i in $(seq 1.0 0.1 1.6)
do
#  echo $i
  python custom/my_own/imgs_processing/imgs_zero_padding.py --ratio ratio_$i
done