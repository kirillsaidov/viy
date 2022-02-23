import os
import glob
import pathlib
import numpy as np

# CNN
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from yolov5model import getDevice

train_data_path = '../data/age/train'
test_data_path = '../data/age/test'

# get device: cpu or cuda(gpu)
device = getDevice()

# create a data transformation variable
imgPrep = transforms.Compose([
	# resize image
	transforms.Resize((128, 128)),

	# add variation to our dataset by flipping it horizontaly
	transforms.RandomHorizontalFlip(0.5),

	# change color value of each pixel from [0;255] to [0;1] and change image data type from numpy array to tensor
	# PyTorch framework expects tensor as input
	transforms.ToTensor(),

	# normalize tensors from [0;1] to [-1;1]
	# formula used to calculate new pixel values: z = (x - mean)/std, where
	# x - old pixel value
	# z - new pixel value
	# this is basically std normal distribution
	transforms.Normalize(
		mean = [0.5, 0.5, 0.5], # mean of [r, g, b] values
		std = [0.5, 0.5, 0.5] 	# std deviation of [r, g, b] values
	),
])

# create a dataloader to load and transform the data automatically
trainDataLoader = DataLoader(
	# where to get the data and transform it
	torchvision.datasets.ImageFolder(train_data_path, transform = imgPrep),

	# depends on your machine's cpu or gpu memory,
	# if size is higher than your memory, it may lead to memory overload and error
	batch_size = 2,

	# randomize images
	shuffle = True
)

testDataLoader = DataLoader(
	# where to get the data and transform it
	torchvision.datasets.ImageFolder(test_data_path, transform = imgPrep),

	# depends on your machine's cpu or gpu memory,
	# if size is higher than your memory, it may lead to memory overload and error
	batch_size = 2,

	# randomize images
	shuffle = True
)

# get folder names which will be our classes
root = pathlib.Path(train_data_path)
classes = sorted([i.name.split('/')[-1] for i in root.iterdir()])

# remove folders or files starting with a dot '.'
for i in classes:
	if i.startswith("."):
		classes.remove(i)

print(classes)

# creating a CNN model class
class CNNModel(nn.Module):
	def __init__(self, number_of_classes = 10):
		super(CNNModel, self).__init__()

		# specifying all layers in our network
















#
