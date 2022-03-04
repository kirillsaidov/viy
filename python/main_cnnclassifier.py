import torch
from torchvision.transforms import transforms

import numpy as np
import matplotlib.pyplot as plt

import cnnclassifier as cnn
from yolov5model import getDevice

# TRAINING 

train_path = '../data/gender/train'
val_path = '../data/gender/val'
epochs = 28
batch_size = 48
lr = 0.05

device = torch.device(getDevice())
model = cnn.CNNClassifier(
    num_classes = 2, 
    batch_size = batch_size,
    img_width = 64,
    img_height = 84,
    #img_norm_mean = [0.63154647, 0.48489257, 0.41346439],
    #img_norm_std = [0.21639832, 0.19404103, 0.18550038]
    img_norm_mean = [0.65625078, 0.48664141, 0.40608295],
    img_norm_std = [0.20471508, 0.17793475, 0.16603905]
).to(device)

val_acc, loss = cnn.train(
    model = model, 
    train_path = train_path, 
    val_path = val_path, 
    epochs = epochs,
    learning_rate = lr
)

plt.plot(val_acc)
plt.xlabel('epochs')
plt.ylabel('validation accuracy')
plt.show()

plt.plot(loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# PREDICTING
"""
pred_path = '../data/age/pred'
transformer = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.5, 0.5, 0.5],
        std = [0.5, 0.5, 0.5]
    )
])

model = cnn.loadModel("best.pt")
pred = cnn.predict(model, pred_path, transformer, "classes.txt")

for (key, value) in pred.items():
    print(f"{key}\t: {value}")

"""









