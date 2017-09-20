#!/bin/sh
while true
do
python ../cifar10_on_resnet/cifar10_eval.py >> eval_result.txt
sleep 60
done
