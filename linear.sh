
nohup \
python linear.py \
--model resnet18 \
--dataset cifar10 \
--size 64 \
--distortion supcon \
--device cuda:0 \
--cosine \
--warm \
--learning_rate 0.5 \
--epochs 200 \
--test_epoch 1 \
--test_dataset cifar10 \
> linear.out

