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
from torch.autograd import Variable
from yolov5model import getDevice

train_data_path = '../data/age/train'
test_data_path = '../data/age/test'

img_width = 128
img_height = 128
batch_s = 12

# get device: cpu or cuda(gpu)
device = torch.device(getDevice())

# create a data transformation variable
imgPrep = transforms.Compose([
    # resize image
    transforms.Resize((img_width, img_height)),

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
        std = [0.5, 0.5, 0.5]   # std deviation of [r, g, b] values
    ),
])

# create a dataloader to load and transform the data automatically
trainDataLoader = DataLoader(
    # where to get the data and transform it
    torchvision.datasets.ImageFolder(train_data_path, transform = imgPrep),

    # depends on your machine's cpu or gpu memory,
    # if size is higher than your memory, it may lead to memory overload and error
    batch_size = batch_s,

    # randomize images
    shuffle = True
)

testDataLoader = DataLoader(
    # where to get the data and transform it
    torchvision.datasets.ImageFolder(test_data_path, transform = imgPrep),

    # depends on your machine's cpu or gpu memory,
    # if size is higher than your memory, it may lead to memory overload and error
    batch_size = batch_s,

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

        """
            Specifying all layers in our network
                * input shape = (batch_size, number of channels (rgb = 3), img_width, img_height)
                * output size after convolution: (width - kernel_size + 2 * padding) / stride + 1
                    - kernel is our filter
                    - stride is how much we need to move (in our case we move 1 element at a time)
                    - padding adds space around your input data (image) in order to avoid output size reduction 
                      as we do element-wise matrix multiplication (with our filter)
        """
        
        kernel_size = 3
        padding = 1
        stride = 1
        
        # layer 1
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 3, 

                # out_channels are determined by the user,
                # I couldn't find any info on what value to choose for out_channel,
                # I will have to test the model, to see what suits best.
                out_channels = 12, 
                kernel_size = kernel_size, 
                stride = stride, 
                padding = padding
            ),
            nn.BatchNorm2d(
                # num_features = out_channels
                num_features = 12,
            ),
            nn.ReLU(),

            # Pooling is basically a process of 'downscaling' the image 
            # obtained from a previous layer. There are 4 pooling methods:
            #   - Max pooling
            #   - Average pooling
            #   - Global max pooling
            #   - Global average pooling

            # In this example, we will use the Max Pooling method. 

            # How it works:
            #   We have our kernel matrix  slide through the image as with the 
            #   convolutional layer, but his time, instead of calculating the
            #   element-wise multiplication, we take the obtain the max value.
            #   The default value for stride is usually the kernel size.
            nn.MaxPool2d(kernel_size = 2) # stride = kernel_size by default
        )

        # layer 2
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 12,           # in the previous layer out_channel = 12 
                out_channels = 24, 
                kernel_size = kernel_size, 
                stride = stride, 
                padding = padding
            ),
            nn.ReLU()
        )
        
        # layer 3
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 24,           # in the previous layer out_channel = 24 
                out_channels = 36, 
                kernel_size = kernel_size, 
                stride = stride, 
                padding = padding
            ),
            nn.BatchNorm2d(
                num_features = 36,
            ),
            nn.ReLU()
        )
        
        # final layer (in_features = final_channels * final_img_width * final_img_height)
        #   we divide by 4 after the Max Polling we get the image of size: (img_width/2, img_height/2)
        self.fully_connected_layer = nn.Linear(
            in_features = int(36 * img_width * img_height / 4),
            out_features = int(number_of_classes)
        )
        

    """
    Detect data class (feed forward function)
    """
    def forward(self, X):
        out = self.conv_layer_1(X)
        out = self.conv_layer_2(out)
        out = self.conv_layer_3(out)

        # the output will be in the matrix form of size:
        #   (batch_size, final_channels, final_img_width, final_img_height)
        
        # reshaping the matrix into vector of data
        out = out.view(out.size(0), -1)
        # out = out.view(-1, int(36 * img_width * img_height / 4))
        
        # feed the data into fully_connected_layer
        out = self.fully_connected_layer(out)

        return out


# create a model and send it to device
model = CNNModel(len(classes)).to(device)

# preparing before training
epochs = 3
learning_rate = 0.05
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.01)

#calculating the size of training and testing images
img_train_count = len(glob.glob(train_data_path + '/**/*.jpg'))
img_test_count = len(glob.glob(test_data_path + '/**/*.jpg'))

print(f"Train, test size: {img_train_count}, {img_test_count}")

# training the model
best_accuracy = 0
for epoch in range(epochs):
    # ---------- TRAINING DATASET. Training and evaluation. ---------- 
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

    # ------------------ TEST DATASET. Evaluation. -------------------
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

    # ----------------------- MANAGE RESULTS ------------------------
    print(f'Epoch: {epoch}')
    print(f'Train loss: {train_loss}')
    print(f'Train accuracy: {train_accuracy}')
    print(f'Test accuracy: {test_accuracy}\n')

    # save the best model
    if test_accuracy > best_accuracy:
        model_scripted = torch.jit.script(model)
        model_scripted.save('best_model.pt')
        best_accuracy = test_accuracy
















#
