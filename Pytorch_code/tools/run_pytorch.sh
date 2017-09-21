#!/bin/sh

python ../cifar10_on_resnet/cifar10_on_resnet.py --lr=0.1 \
--epochs=450 \
--momentum=0.9 \
--test-batch-size=1000 \
--enable-gpu=True\


