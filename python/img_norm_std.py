import glob
import cv2 as cv
import numpy as np

# img = cv.imread("../data/age/train/6-14/1953.jpg")
# mean, std = cv.meanStdDev(img)

def cal_mean_std(img_path):
    imgs = glob.glob(img_path + '/*.jpg')
    if not imgs:
        print("No images found! Exiting...")
        return
    else:
        print(f'{len(imgs)} found!')

    mean = []
    std = []
    for i in imgs:
        img = cv.imread(i)
        m, s = cv.meanStdDev(img)

        mean.append(m.reshape((3,)))
        std.append(s.reshape((3,)))


    m_array = np.array(mean)
    s_array = np.array(std)
    mean = m_array.mean(axis=0, keepdims=True)/255
    std = s_array.mean(axis=0, keepdims=True)/255

    print('mean: ', mean[0][::-1])
    print('std:  ', std[0][::-1])

cal_mean_std("../data/age/**/**/")