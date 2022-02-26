"""
README: IMPORTANT!

    BEFORE USING THE MODEL MODIFY THE createLayers() AND forward() FUNCTIONS. ADD OR REMOVE LAYERS TO YOUR LIKING, OR USE THE DEFAULTS. IN THAT CASE THERE IS NOTHING TO CHANGE.
"""

import sys
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

# warnings
import warnings
warnings.filterwarnings("ignore")

# CNN classifier for image calssification
class CNNClassifier(nn.Module):
    """
    Initializes the class
        num_classes = number of classes in a model
        batch_size = number of images send to the network at once (depends on available memory)
        transformer = data transformer from torchvision.transforms (if None, default is used)
        img_width = image width
        img_height = image height
        kernel_size = filter matrix in the convolutional layer (3 by default)
        padding = add padding to an image (1 by default)
        stride = matrix sliding speed (1 by default)
        random_horizontal_flip = adding variety from 0 to 1 (0.5 by default)
        img_norm_mean = mean of [r, g, b] values of images ([0.5, 0.5, 0.5] by default)
        img_norm_std = std deviation of [r, g, b] values of images ([0.5, 0.5, 0.5] by default)
        img_channels = number of channels in an image ([r, g, b] = 3 by default)
    """
    def __init__(self,
        num_classes = 1,
        batch_size = 16,
        transformer = None,
        img_width = 128,
        img_height = 128,
        kernel_size = 3,
        padding = 1,
        stride = 1,
        random_horizontal_flip = 0.5,
        img_norm_mean = [0.5, 0.5, 0.5],
        img_norm_std = [0.5, 0.5, 0.5],
        img_channels = 3
    ):
        # initialize the constructor
        super(CNNClassifier, self).__init__()

        # save variables
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.img_channels = img_channels
        self.img_width = img_width
        self.img_height = img_height

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

        self.createLayers()

    """
    Creates CNN layers
    """
    def createLayers(self):
        self.layer_1 = self.__addLayerType1()
        self.layer_2 = self.__addLayerType2()
        self.layer_3 = self.__addLayerType2()
        self.layer_4 = self.__addLayerType3()
        self.layer_5 = self.__addLayerType1()
        self.layer_6 = self.__addLayerFC()

    """
    Feed forward function
    """
    def forward(self, X):
        out = self.layer_1(X)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)

        # reshaping the matrix into vector of data
        out = out.view(out.size(0), -1)

        # feed the data into fully_connected_layer
        out = self.layer_6(out)

        return out

    """
    Exports the model to TorchScript type
    """
    def exportModel(self, model_name = "last.pt"):
        model_scripted = torch.jit.script(self)
        model_scripted.save(model_name)

    """
    Load the TorchScript model
    """
    def loadModel(self, model_name = None):
        if model_name:
            print("Load model.")
        else:
            print("Model name is not provided; nothing to load.")

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
        self.img_channels = int(self.img_channels * outMultFactor)
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
                kernel_size = self.kernel_size,
                stride = self.stride,
                padding = self.padding
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
                kernel_size = self.kernel_size,
                stride = self.stride,
                padding = self.padding
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

"""
Trains the model
    train_path = path to the train data folder (each category in its own folder)
    test_path = path to the test data folder (each category in its own folder)
    epochs = number of epochs to run
    learning_rate = learning rate of the model
    weight_decay = weight decay when learning to avoid overfitting
    model_name = custom model name (best.pt)
    export_model = export the model or not
    verbose = verbosity to print every step
"""
def train(model,
    train_path = None,
    test_path = None,
    epochs = None,
    learning_rate = 0.5,
    weight_decay = 0.01,
    model_name = "best.pt",
    export_model = True,
    verbose = True,
):
    assert train_path, "ERROR: train path not provided! Exiting..."
    assert test_path, "ERROR: test path not provided! Exiting..."

    # get folder names which will be our classes
    root = pathlib.Path(train_path)
    classes = sorted([i.name.split('/')[-1] for i in root.iterdir()])

    # remove folders or files starting with a dot '.'
    for i in classes:
        if i.startswith("."):
            classes.remove(i)
    num_classes = len(classes)

    assert model.num_classes == num_classes, f"Number of classes specified ({model.num_classes}) is not the same as classes found ({num_classes})."

    if verbose:
        print(f"==> {num_classes} classes found.")
        print("==> Creating data loaders...")

    # create a dataloader to load and transform the data automatically
    trainDataLoader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform = model.transformer),
        batch_size = model.batch_size,
        shuffle = True
    )

    testDataLoader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform = model.transformer),
        batch_size = model.batch_size,
        shuffle = True
    )

    # calculating the size of training and testing images
    img_train_count = len(glob.glob(train_path + '/**/*.jpg'))
    img_test_count = len(glob.glob(test_path + '/**/*.jpg'))

    # create loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    if verbose:
        print("==> Data loaders created...")
        print(f"==> Number of train/test imgs: {img_train_count}/{img_test_count}")
        print("==> Loss function initialized: CrossEntropyLoss")
        print("==> Optimizer initialized: Adam optimizer")
        print("==> Training and evaluation started...\n")

    # ---------------- MODEL TRAINING AND EVALUATION ----------------
    best_accuracy = 0.0
    for epoch in range(0, epochs):
        # ---------------------- TRAINING ---------------------------
        if verbose:
            print(f"==> Training epoch {epoch}/{epochs}:")

        model.train()

        train_accuracy = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(trainDataLoader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # zero out the gradients at the start of a new batch
            optimizer.zero_grad()

            # detect
            outputs = model(images)

            # calculate loss
            loss = loss_function(outputs, labels)

            # back propagation
            loss.backward()

            # update weights and bias
            optimizer.step()

            train_loss += loss.cpu().data * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            # sum accuracy
            train_accuracy += int(torch.sum(prediction == labels.data))

        # calculate accuracy and loss
        train_accuracy = train_accuracy / img_train_count
        train_loss = train_loss / img_train_count

        # ----------------------- EVALUATION ------------------------
        if verbose:
            print(f"\tEvaluating:")

        model.eval()

        test_accuracy = 0.0
        for i, (images, labels) in enumerate(testDataLoader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # get prediction
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            test_accuracy += int(torch.sum(prediction == labels.data))

        test_accuracy = test_accuracy / img_test_count

        # --------------------- MANAGE RESULTS ----------------------
        if verbose:
            print(f"\t---> TrainLoss:\t{train_loss:.3f}")
            print(f"\t---> TrainAccr:\t{train_accuracy:.3f}")
            print(f"\t---> TestAccr: \t{test_accuracy:.3f}")

        # save the best model
        if export_model and test_accuracy > best_accuracy:
            if verbose:
                print("==> Saving the model...")

            model.exportModel(model_name)
            best_accuracy = test_accuracy

    # --------------- END  OF TRAINING AND EVALUATION ---------------
