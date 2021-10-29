#!/usr/bin/env python3
import os
import sys
import cv2 as cv
import numpy as np

# default values
imgs_folder = ''
output_folder = ''
width = 640
height = 640

# get arguments
argv = sys.argv[1:]
if len(argv) < 2:
    print("\n#imgresize: imgresize.py [imgs_folder] [output_folder] {width} {height}\n")
    sys.exit()
elif len(argv) == 2:
    imgs_folder = argv[0]
    output_folder = argv[1]
elif len(argv) == 3:
    imgs_folder = argv[0]
    output_folder = argv[1]
    width = argv[2]
else:
    imgs_folder = argv[0]
    output_folder = argv[1]
    width = argv[2]
    height = argv[3]

# find all images
imgs = [i for i in os.listdir(os.path.join(".", imgs_folder)) if '.jpg' in i or '.jpeg' in i or '.JPEG' in i or '.JPG' in i]

newSize = (int(width), int(height))
for i in imgs:
    # open image
    img = cv.imread(os.path.join(".", imgs_folder, i))

    # resize image
    img = cv.resize(img, newSize)

    # save image
    cv.imwrite(os.path.join(".", output_folder, i), img)
    
    # verbose output
    print(f"#imgresize: <{i}> converted!")
    
#print(os.path.join(".", imgs_folder, imgs[0]))

