"""
README: IMPORTANT!

    BEFORE USING THE MODEL MODIFY THE createLayers() AND forward() FUNCTIONS.
    ADD OR REMOVE LAYERS TO YOUR LIKING, OR USE THE DEFAULTS.
    IN THAT CASE THERE IS NOTHING TO CHANGE.
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
        random_rotation = rotate images randomly in degrees
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
        random_rotation = 30,
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
                transforms.RandomHorizontalFlip(random_horizontal_flip),
                transforms.RandomRotation(random_rotation),
                transforms.RandomPerspective(),
                transforms.RandomGrayscale(p = 0.5),
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
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
        self.layer_1 = self.__addLayerConv__(2)
        self.layer_2 = self.__addLayerConv__(1)
        self.layer_3 = self.__addLayerConv__(1)
        self.layer_4 = self.__addLayerMaxPool__()

        self.layer_5 = self.__addLayerConv__(2)
        self.layer_6 = self.__addLayerConv__(1)
        self.layer_7 = self.__addLayerMaxPool__()

        self.layer_8 = self.__addLayerConv__(2)
        self.layer_9 = self.__addLayerConv__(1)
        self.layer_10 = self.__addLayerMaxPool__()
        """
        self.layer_11 = self.__addLayerConv__(2)
        self.layer_12 = self.__addLayerConv__(1)
        self.layer_13 = self.__addLayerConv__(1)
        self.layer_14 = self.__addLayerMaxPool__()
        self.layer_15 = self.__addLayerConv__(2)
        self.layer_16 = self.__addLayerConv__(1)
        self.layer_17 = self.__addLayerConv__(1)
        self.layer_18 = self.__addLayerMaxPool__()
        """
        self.layer_fc = self.__addLayerClassifier__()

    """
    Feed forward function
    """
    def forward(self, X):
        out = self.layer_1(X)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)

        out = self.layer_5(out)
        out = self.layer_6(out)
        out = self.layer_7(out)

        out = self.layer_8(out)
        out = self.layer_9(out)
        out = self.layer_10(out)

        """
        out = self.layer_11(out)
        out = self.layer_12(out)
        out = self.layer_13(out)
        out = self.layer_14(out)
        out = self.layer_15(out)
        out = self.layer_16(out)
        out = self.layer_17(out)
        out = self.layer_18(out)
        """
        # reshaping the matrix into vector of data
        # out = out.view(out.size(0), -1)

        # feed the data into fully_connected_layer
        out = self.layer_fc(out)

        return out

    """
    Exports the model to TorchScript type
    """
    def exportModel(self, model_name = "last.pt"):
        model_scripted = torch.jit.script(self)
        model_scripted.save(model_name)

    """
    [private] Adds a convolutional layer
    """
    def __addLayerConv__(self, outMultFactor = 2):
        # create a new convolutional layer
        conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels = self.img_channels,           # in the previous layer out_channel = 12
                out_channels = int(self.img_channels * outMultFactor),
                kernel_size = self.kernel_size,
                stride = self.stride,
                padding = self.padding
            ),
            nn.BatchNorm2d(
                num_features = int(self.img_channels * outMultFactor)
            ),
            nn.ReLU(),
        )

        # update values
        self.img_channels = int(self.img_channels * outMultFactor)

        return conv_layer

    """
    [private] Adds a max pool layer
    """
    def __addLayerMaxPool__(self, kernel_size = 2):
        pool_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size = kernel_size, stride = kernel_size),
        )

        self.img_width = self.img_width/kernel_size
        self.img_height = self.img_height/kernel_size

        return pool_layer

    """
    [private] Adds a fully connected layer
    """
    def __addLayerClassifier__(self):
        # in-out features data
        in_out_1 = (
            int(self.img_channels * self.img_width * self.img_height),
            int(self.img_channels * self.img_width * self.img_height / 4)
        )
        in_out_2 = (in_out_1[1], int(in_out_1[1] / 2))
        in_out_3 = (in_out_2[1], self.num_classes)

        classifier_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features = in_out_1[0],
                out_features = in_out_1[1]
            ),
            nn.BatchNorm1d(num_features = in_out_1[1]),
            nn.ReLU(),
            nn.Linear(
                in_features = in_out_2[0],
                out_features = in_out_2[1]
            ),
            nn.BatchNorm1d(num_features = in_out_2[1]),
            nn.ReLU(),
            nn.Linear(
                in_features = in_out_3[0],
                out_features = in_out_3[1]
            ),
            nn.Softmax(dim = 1)
        )

        return classifier_layer

"""
Trains the model
    train_path = path to the train data folder (each category in its own folder)
    val_path = path to the validation data folder (each category in its own folder)
    epochs = number of epochs to run
    optim = optimizer function [
        'Adam', 'RAdam', 'SGD',
        'ASGD', 'Adagrad', 'Adadelta',
        'AdamW', 'Adamax', 'RMSProp'
    ]
    learning_rate = learning rate of the model
    weight_decay = weight decay when learning to avoid overfitting
    momentum = a degree of moving average that denoises the data
    alpha = smoothing constant (power for eta update in case of ASGD model)
    lambd = decay term (ASGD model)
    t0 = point at which to start averaging (ASGD model)
    amsgrad = helps in deciding whether AMSgrad algorithm will be used or not
    centered = the gradient is normalized by an estimation of its variance
    model_name = custom model name (best.pt)
    export_model = export the model or not
    verbose = verbosity to print every step
"""
def train(model,
    train_path = None,
    val_path = None,
    epochs = None,
    optim = 'Adam',
    learning_rate = 0.05,
    weight_decay = 0.01,
    momentum = 0.9,
    alpha = 0.75,
    lambd = 0.0001,
    t0 = 1000000.0,
    amsgrad = False,
    centered = False,
    model_name = "best.pt",
    export_model = True,
    verbose = True,
):
    assert train_path, "ERROR: train path not provided! Exiting..."
    assert val_path, "ERROR: validation path not provided! Exiting..."

    # get folder names which will be our classes
    root = pathlib.Path(train_path)
    classes = sorted([i.name.split('/')[-1] for i in root.iterdir()])

    # remove folders or files starting with a dot '.'
    for i in classes:
        if i.startswith("."):
            classes.remove(i)

    # save classes
    num_classes = len(classes)
    with open("classes.txt", "w") as f:
        for i in range(0, num_classes - 1):
            f.write(classes[i] + ";")
        f.write(classes[num_classes - 1])

    assert model.num_classes == num_classes, f"Number of classes specified ({model.num_classes}) is not the same as classes found ({num_classes})."

    if verbose:
        print(f"==> Device: {getDevice()}")
        print(f"==> {num_classes} classes found.")
        print(f"==> Classes saved to classes.txt")
        print("==> Creating data loaders...")

    # create a dataloader to load and transform the data automatically
    trainDataLoader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform = model.transformer),
        batch_size = model.batch_size,
        shuffle = True,
        drop_last = True,
    )

    valDataLoader = DataLoader(
        torchvision.datasets.ImageFolder(val_path, transform = model.transformer),
        batch_size = model.batch_size,
        shuffle = True,
        drop_last = True,
    )

    # calculating the size of training and validation images
    img_train_count = len(glob.glob(train_path + '/**/*.jpg'))
    img_val_count = len(glob.glob(val_path + '/**/*.jpg'))

    # create loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = None
    if optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    if optim == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
    elif optim == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr = learning_rate, lambd = lambd, alpha = alpha, t0 = t0)
    elif optim == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    elif optim == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay, amsgrad = amsgrad)
    elif optim == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    elif optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate, weight_decay = weight_decay, momentum = momentum, centered = centered, alpha = alpha)

    if verbose:
        print("==> Data loaders created...")
        print(f"==> Number of train/val imgs: {img_train_count}/{img_val_count}")
        print("==> Loss function initialized: CrossEntropyLoss")
        print("==> Optimizer initialized: {optim} optimizer")
        print("==> Training and evaluation started...\n")

    # ---------------- MODEL TRAINING AND EVALUATION ----------------
    tloss_list = []
    val_acc_list = []
    best_accuracy = 0.0
    for epoch in range(0, epochs):
        # ---------------------- TRAINING ---------------------------
        if verbose:
            print(f"==> Training epoch {epoch}/{epochs - 1}:")

        model.train()

        train_accuracy = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(trainDataLoader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

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
        tloss_list.append(train_loss)

        # ----------------------- EVALUATION ------------------------
        if verbose:
            print(f"\tEvaluating:")

        model.eval()

        val_accuracy = 0.0
        for i, (images, labels) in enumerate(valDataLoader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # get prediction
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            val_accuracy += int(torch.sum(prediction == labels.data))

        val_accuracy = val_accuracy / img_val_count
        val_acc_list.append(val_accuracy)

        # --------------------- MANAGE RESULTS ----------------------
        if verbose:
            print(f"\t---> train_loss: {train_loss:.3f}")
            print(f"\t---> train_acc:  {train_accuracy:.3f}")
            print(f"\t---> val_acc:   {val_accuracy:.3f}")

        # save the best model
        if export_model and val_accuracy > best_accuracy:
            if verbose:
                print("==> Saving the model...")

            model.exportModel(model_name)
            best_accuracy = val_accuracy

    return val_acc_list, tloss_list

    # --------------- END  OF TRAINING AND EVALUATION ---------------

"""
Loads the TorchScript model
"""
def loadModel(model_load_path = None):
    assert model_load_path, "ERROR: model load path not provided! Exiting..."

    # load a torchscript model
    model = torch.jit.load(model_load_path)
    model.eval()

    return model

"""
Predicts image class
    verbose = verbosity to print every step
"""
def predict(model = None, img_path = None, transformer = None, classes_path = None, verbose = True):
    assert model, "ERROR: model argument not provided! Exiting..."
    assert img_path, "ERROR: imgs path not provided! Exiting..."
    assert transformer, "ERROR: data transformer not provided! Exiting..."
    assert classes_path, "ERROR: path to classes file not provided! Exiting..."

    # find all imgs in the directory, if a single img is given, convert it into an array
    # and continue
    if '.jpg' in img_path:
        img_path = [img_path]
    else:
        img_path = glob.glob(img_path + '/*.jpg')

    # read classes from a file
    classes = []
    with open(classes_path, "r") as f:
        lines = [line.rstrip() for line in f]
        classes = lines[0].split(";")

    if verbose:
        print(f"==> Found {len(img_path)} images...")
        print(f"==> Loading {classes_path}")
        print(f"==> Loaded: {classes}")

    # predict
    pred = []
    for i in img_path:
        if verbose:
            print(f"Predicting: {i}")

        # load image
        img = Image.open(i)

        # transform data
        img_tensor = transformer(img).float()

        # PyTorch treats all images as batches. We need to insert an extra batch dimension.
        img_tensor = img_tensor.unsqueeze_(0)

        # send images to GPU if available
        if torch.cuda.is_available():
            img_tensor.cuda()

        # predict
        out = model(img_tensor)

        # get the class with the maximum probability
        class_id = out.data.numpy().argmax()

        # get class name
        pred.append({'in': i, 'out': classes[class_id]})

    if verbose:
        print(f"==> DONE.\n")

    return pred
