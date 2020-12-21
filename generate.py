import cv2
import numpy as np
import random
import copy

def quantization(pixels, bins,range_):
    m = range_[0]
    interval_size = range_[1]-range_[0]
    interval_size/=bins

    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            pixels[i][j] = ((pixels[i][j]-m)/interval_size)

           
    return pixels

def haze(img):
    shape = img.shape
    d_map = np.random.random(shape)

    k = int(random.uniform(0.85,1)*255)
    A = np.array([k,k,k])

    beta = 1

    t = np.exp(-beta*d_map)

    hazed_image = copy.deepcopy(img).astype(float)

    for i in range(hazed_image.shape[0]):
        for j in range(hazed_image.shape[1]):
            hazed_image[i][j] = img[i][j]*t[i][j] + A*(1-t[i][j])
            # print(hazed_image[i][j],img[i][j]*t[i][j],A*(1-t[i][j]))

    return img,d_map,quantization(hazed_image,256,[np.min(hazed_image),np.max(hazed_image)])
name = "img2"

image = cv2.imread("./train/"+name+".jpg")

image,depth_map,hazed_image = haze(image)

cv2.imwrite("./train/"+name+"_h.jpg",hazed_image)
cv2.imwrite("./train/"+name+"_d.jpg",depth_map*255)

