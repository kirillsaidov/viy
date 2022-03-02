import torch
from torchvision.transforms import transforms

import cnnclassifier as cnn
from yolov5model import getDevice

# TRAINING 

train_path = '../data/age/train'
test_path = '../data/age/test'
epochs = 64

device = torch.device(getDevice())
model = cnn.CNNClassifier(num_classes = 3, batch_size = 12).to(device)
cnn.train(model = model, train_path = train_path, test_path = test_path, epochs = epochs)


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









