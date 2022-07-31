import torch
from torchvision.transforms import transforms

import numpy as np
import matplotlib.pyplot as plt

import cnnclassifier as cnn
from yolov5model import getDevice

device = torch.device(getDevice())

# TRAINING

"""
optim = optimizer function [
    'Adam', 'RAdam', 'SGD',
    'ASGD', 'Adagrad', 'Adadelta',
    'AdamW', 'Adamax', 'RMSProp'
]

ASGD, SGD and Adadelta are the best optimizers.
"""

# TRAINING
train_path = '../data/data_gender/train'
val_path = '../data/data_gender/val'
epochs = 7
batch_size = 32
lr = 0.08
optimizer = 'ASGD'

model = cnn.CNNClassifier(
    num_classes = 2,
    batch_size = batch_size,
    img_width = 28,
    img_height = 28,
    # age
    # img_norm_mean = [0.63154647, 0.48489257, 0.41346439],
    # img_norm_std = [0.21639832, 0.19404103, 0.18550038]
    # gender
    img_norm_mean = [0.65625078, 0.48664141, 0.40608295],
    img_norm_std = [0.20471508, 0.17793475, 0.16603905]
).to(device)

val_acc, loss = cnn.train(
    model = model,
    train_path = train_path,
    val_path = val_path,
    epochs = epochs,
    optim = optimizer,
    momentum = 0.7,
    alpha = 0.7,
    lambd = 0.0001,
    t0 = 1000000.0,
    learning_rate = lr,
    model_name = "gender_model_tiny.pt",
)

#plt.plot(val_acc)
#plt.xlabel('epochs')
#plt.ylabel('validation accuracy')
#plt.show()

#plt.plot(loss)
#plt.xlabel('epochs')
#plt.ylabel('loss')
#plt.show()

# _____________________________________________________________

# PREDICTING
"""
import cv2
from PIL import Image

pred_img_path = '../data/archive/female.jpg'

# transformers
transformer_gender = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.65625078, 0.48664141, 0.40608295],
        [0.20471508, 0.17793475, 0.16603905],
    ),
])

transformer_age = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

model_gender = cnn.loadModel("VIY/weights/gender_model89.pt").to(device)
# model_age = cnn.loadModel("VIY/weights/age_model521.pt").to(device)

# read img
img = cnn.img2array(cv2.imread(pred_img_path))
classes = cnn.readClasses("VIY/weights/gender_classes.txt")

# predict
pred = cnn.predict(model_gender, img, transformer_age, classes)
print(pred)
"""
