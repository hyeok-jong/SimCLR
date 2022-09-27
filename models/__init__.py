from .resnet import *
from .vgg import *
from .alex import *




# Note that Pytorch uses avgpooling layer for react any size images.
# For instance ResNet, doing so, shape after feature extractor is 1*1

# https://github.com/HobbitLong/SupContrast/blob/master/networks/resnet_big.py
# https://github.com/sthalles/SimCLR/blob/master/models/resnet_simclr.py
# Both repos have avgpooling
# So I did so