import torch
from torchvision.transforms import transforms

import cnnclassifier as cnn
from yolov5model import getDevice

# TRAINING 

train_path = '../data/age/train'
test_path = '../data/age/test'
epochs = 32
batch_size = 8
lr = 0.05

device = torch.device(getDevice())
model = cnn.CNNClassifier(
    num_classes = 3, 
    batch_size = batch_size,
    img_width = 128,
    img_height = 128,
    img_norm_mean = [0.63154647, 0.48489257, 0.41346439],
    img_norm_std = [0.21639832, 0.19404103, 0.18550038]
).to(device)

cnn.train(
    model = model, 
    train_path = train_path, 
    test_path = test_path, 
    epochs = epochs,
    learning_rate = lr
)


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









