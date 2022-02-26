import torch
import cnnclassifier as cnn
from yolov5model import getDevice

train_path = '../data/age/train'
test_path = '../data/age/test'

device = torch.device(getDevice())

model = cnn.CNNClassifier(num_classes = 3).to(device)
cnn.train(model = model, train_path = train_path, test_path = test_path, epochs = 21)
