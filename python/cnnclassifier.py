import glob
import pathlib
import cv2
import numpy as np
from io import open
from PIL import Image

# PyTorch
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torchvision.models import squeezenet1_1
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.functional as F

# yolov5 model
from yolov5model import getDevice

# CNN classifier for image calssification
class CNNClassifier(nn.Module):
    """
    Initializes the class
        train_path = path to the train data folder (each category in its own folder)
        test_path = path to the test data folder (each category in its own folder)
        batch_size = number of images send to the network at once (depends on available memory)
        transformer = data transformer from torchvision.transforms (if None, default is used)
        img_width = image width
        img_height = image height
        kernel_size = filter matrix in the convolutional layer (3 by default)
        padding = add padding to an image (1 by default)
        stride = matrix sliding speed (1 by default)
        layers = number of layers in the CNN (9 by default)
        random_horizontal_flip = adding variety from 0 to 1 (0.5 by default)
        img_norm_mean = mean of [r, g, b] values of images ([0.5, 0.5, 0.5] by default)
        img_norm_std = std deviation of [r, g, b] values of images ([0.5, 0.5, 0.5] by default)
        img_channels = number of channels in an image ([r, g, b] = 3 by default)
    """
    def __init__(self,
        train_path = None,
        test_path = None,
        batch_size = 16,
        transformer = None,
        img_width = 128,
        img_height = 128,
        kernel_size = 3,
        padding = 1,
        stride = 1,
        layers = 9,
        random_horizontal_flip = 0.5,
        img_norm_mean = [0.5, 0.5, 0.5],
        img_norm_std = [0.5, 0.5, 0.5],
        img_channels = 3
    ):
        # initialize the constructor
        super(CNNClassifier, self).__init__()

        # save variables
        self.layers = layers
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.img_channels = img_channels
        self.img_width = img_width
        self.img_height = img_height

        # get folder names which will be our classes
        if train_path:
            root = pathlib.Path(train_path)
            classes = sorted([i.name.split('/')[-1] for i in root.iterdir()])

            # remove folders or files starting with a dot '.'
            for i in classes:
                if i.startswith("."):
                    classes.remove(i)

        self.num_classes = len(classes)

        # create a data transformer
        if transformer == None:
            self.transformer = transforms.Compose([
                transforms.Resize((img_width, img_height)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = img_norm_mean,
                    std = img_norm_std
                )
            ])
        else:
            self.transformer = transformer

        # create a dataloader to load and transform the data automatically
        if train_path:
            self.trainDataLoader = DataLoader(
                torchvision.datasets.ImageFolder(train_path, transform = self.transformer),
                batch_size = batch_size,
                shuffle = True
            )

        if test_path:
            self.testDataLoader = DataLoader(
                torchvision.datasets.ImageFolder(test_path, transform = self.transformer),
                batch_size = batch_size,
                shuffle = True
            )

        self.layers = self.__createLayers()

    """
    Feed forward function
    """
    def forward(self, X):
        out = self.layers[0](X)

        nlayers = len(self.layers)
        for i in range(1, nlayers - 1):
            out = self.layers[i](out)

        # reshaping the matrix into vector of data
        out = out.view(out.size(0), -1)

        # feed the data into fully_connected_layer
        out = self.layers[nlayers - 1](out)

        return out

    """
    Trains the model
    """
    def train(self, epochs, learning_rate = 0.5, weight_decay = 0.01, verbose = True):
        pass

    """
    Exports the model to TorchScript type
    """
    def exportModel(self, name = "last.pt"):
        pass

    """
    [private] Creates CNN layers
    """
    def __createLayers(self):
        conv_layers = list()

    # 1st layer with max pooling
        self.conv_layers.append(self.__addLayerType1(4))

        for i in range(0, int(layers/2)):
            self.conv_layers.append(self.__addLayerType2(i))
            self.conv_layers.append(self.__addLayerType3(i))

        for i in range(0, int(layers/2), -1):
            self.conv_layers.append(self.__addLayerType2(1/i))
            self.conv_layers.append(self.__addLayerType3(1/i))

        # last fully connected layer
        conv_layers.append(self.__addLayerFC())

        return conv_layers

    """
    [private] Creates a the 1st layer in CNN
    """
    def __addLayerType1(self, outMultFactor = 4):
        # create a new convolutinal layer
        conv_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels = self.img_channels,
                    out_channels = int(self.img_channels * outMultFactor),
                    kernel_size = self.kernel_size,
                    stride = self.stride,
                    padding = self.padding
                ),
                nn.BatchNorm2d(num_features = int(self.img_channels * outMultFactor)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2)
            )

        # update values
        self.in_channels = int(self.img_channels * outMultFactor)
        self.img_width = self.img_width/2
        self.img_height = self.img_height/2

        return conv_layer

    """
    [private] Creates type 2 layer in CNN
    """
    def __addLayerType2(self, outMultFactor = 2):
        # create a new convolutional layer
        conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels = self.img_channels,           # in the previous layer out_channel = 12
                out_channels = int(self.img_channels * outMultFactor),
                kernel_size = kernel_size,
                stride = stride,
                padding = padding
            ),
            nn.ReLU()
        )

        # update values
        self.img_channels = int(self.img_channels * outMultFactor)

        return conv_layer
    """
    [private] Creates a type 3 layer in CNN
    """
    def __addLayerType3(self, outMultFactor = 2):
        # create a new convolutional layer
        conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels = self.img_channels,           # in the previous layer out_channel = 12
                out_channels = int(self.img_channels * outMultFactor),
                kernel_size = kernel_size,
                stride = stride,
                padding = padding
            ),
            nn.BatchNorm2d(num_features = int(self.img_channels * outMultFactor)),
            nn.ReLU()
        )

        # update values
        self.img_channels = int(self.img_channels * outMultFactor)

        return conv_layer

    """
    [private] Creates a fully connected (final) layer in CNN
    """
    def __addLayerFC(self):
        fc_layer = nn.Linear(
            in_features = int(self.img_channels * self.img_width * self.img_height),
            out_features = int(self.num_classes)
        )

        return fc_layer
