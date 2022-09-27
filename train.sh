

nohup python train.py \
--model resnet18 \
--dataset cifar10 \
--size 64 \
--distortion supcon \
--device cuda:1 \
--cosine \
--warm \
--epochs 1200 > resnet18_train.out &


