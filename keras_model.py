import keras 
import numpy as np 
from keras.applications import vgg16, inception_v3, resnet50, mobilenet

vgg_model = vgg16.VGG16(weights='imagenet')
print(type(vgg_model))

inception_model = inception_v3.InceptionV3(weights='imagenet')
print(type(inception_model))

resnet_model = resnet50.ResNet50(weights='imagenet')
print(type(resnet_model))

mobilenet_model = mobilenet.MobileNet(weights='imagenet')
print(type(mobilenet_model))