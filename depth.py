import cv2 
import numpy as np
import copy 

def quantization(pixels, bins,range_):
    m = range_[0]
    interval_size = range_[1]-range_[0]
    interval_size/=bins

    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            pixels[i][j] = ((pixels[i][j]-m)/interval_size)

           
    return pixels
def visualise(depth_map):
    depth_map = quantization(depth_map,255,[depth_map.min(),depth_map.max()]).astype(np.uint8)

    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_HOT)
    cv2.imshow("Hazy",depth_map)
    cv2.waitKey(0)

def relu(x):
    if x<0:
        return 0
    else:
        return x

def reverse_relu(bound,x):
    if x>bound:
        return bound
    else:
        return x

# Variables
filename = "forest.jpg"

theta0 = 0.121779
theta1 = 0.959710
theta2 = -0.780245
sigma = 0.041337

n_size = 5 # Size of neighbourhood considered for min filter
blur_strength = 15 # Strength of blurring after min filter in depthmap

h_img = cv2.imread(filename)

hsv = cv2.cvtColor(h_img, cv2.COLOR_BGR2HSV)
value = hsv[:,:,2].astype('float')/255
saturation = hsv[:,:,1].astype('float')/255
# print(value)
depth_map = theta0 + theta1*value + theta2*saturation + np.random.normal(0,sigma, hsv[:,:,0].shape)
# visualise(depth_map)

new_depth_map = copy.deepcopy(depth_map)

width = depth_map.shape[0]
height = depth_map.shape[1]
for i in range(width):
    for j in range(height):
        x_low = relu(i-n_size)
        x_high =  reverse_relu(width-1,i+n_size)+1
        y_low = relu(j-n_size)
        y_high =  reverse_relu(height-1,j+n_size)+1
        # print(depth_map[x_low:x_high][y_low].shape)
        new_depth_map[i][j] = np.min( depth_map[x_low:x_high,y_low:y_high] )

visualise(new_depth_map)
new_depth_map = cv2.GaussianBlur(new_depth_map,(blur_strength,blur_strength),0)
visualise(new_depth_map)