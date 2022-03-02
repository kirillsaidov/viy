import glob
import pathlib
from io import open

# working with images
from PIL import Image

# PyTorch
import torch
from torchvision.transforms import transforms
from torchvision.models import squeezenet1_1
from torch.utils.data import DataLoader

img_width = 128
img_height = 128

# get folder names which will be our classes
train_data_path = '../data/age/train'
root = pathlib.Path(train_data_path)
classes = sorted([i.name.split('/')[-1] for i in root.iterdir()])

# remove folders or files starting with a dot '.'
for i in classes:
    if i.startswith("."):
        classes.remove(i)

print(classes)

# ------------------ MAKING PREDICTIONS -------------------

# create a data transformation variable
imgPredPrep = transforms.Compose([
    # resize image
    transforms.Resize((img_width, img_height)),

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

def predict(img_path, transformer):
    # load image
    img = Image.open(img_path)

    # transform data
    img_tensor = transformer(img).float()

    # PyTorch treats all images as batches. We need to insert an extra batch dimension.
    img_tensor = img_tensor.unsqueeze_(0)

    # send images to GPU if available
    #if torch.cuda.is_available():
    #    img_tensor.cuda()    
    
    # predict
    out = model(img_tensor)
    
    # get the class with the maximum probability
    class_id = out.data.numpy().argmax()
    
    # get class name
    className = classes[class_id]
    
    return className

# find all pred images
pred_data_path = '../data/age/pred'
pred_imgs = glob.glob(pred_data_path + '/*.jpg')

# load model with best weights
model = torch.jit.load('best.pt')
model.eval()

pred_dict={}
for i in pred_imgs:
    pred = predict(i, imgPredPrep)
    print(f"Predicted: {i}   \t=> {pred}")

