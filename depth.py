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
filename = "./images/forest.jpg"

theta0 = 0.121779
theta1 = 0.959710
theta2 = -0.780245
sigma = 0.041337

n_size = 5 # Size of neighbourhood considered for min filter
blur_strength = 15 # Strength of blurring after min filter in depthmap

h_img = cv2.imread(filename)

hsv = cv2.cvtColor(h_img, cv2.COLOR_BGR2HSV)
value = hsv[:,:,2].astype('float')/255 # Intensity values of image
saturation = hsv[:,:,1].astype('float')/255 # Saturation values of image

depth_map = theta0 + theta1*value + theta2*saturation + np.random.normal(0,sigma, hsv[:,:,0].shape)

new_depth_map = copy.deepcopy(depth_map) # min filtered depth  map

width = depth_map.shape[1]
height = depth_map.shape[0]

for i in range(height):
    for j in range(width):
        x_low = relu(i-n_size)
        x_high =  reverse_relu(height-1,i+n_size)+1
        y_low = relu(j-n_size)
        y_high =  reverse_relu(width-1,j+n_size)+1
        new_depth_map[i][j] = np.min( depth_map[x_low:x_high,y_low:y_high] )

# visualise(new_depth_map)

blurred_depth_map = cv2.GaussianBlur(new_depth_map,(blur_strength,blur_strength),0) # Gaussian blur of depthmap (d(x))

# visualise(blurred_depth_map)


depth_map_1d = np.ravel(blurred_depth_map)

rankings = np.argsort(depth_map_1d)

threshold = (99.9*len(rankings))/100
indices = np.argwhere(rankings>threshold).ravel()

indices_image_rows = indices//width
indices_image_columns = indices % width

atmospheric_light = np.zeros(3) # A
intensity = -np.inf
for x in range(len(indices_image_rows)):
    i = indices_image_rows[x]
    j = indices_image_columns[x]

    if value[i][j] >= intensity:
        atmospheric_light = h_img[i][j]
        intensity = value[i][j]

beta = 1

# Calculating output image

t = np.exp(-beta*blurred_depth_map)

denom = np.clip(t,0.1,0.9)
# denom = t
numer = h_img.astype("float") - atmospheric_light.astype("float")

output_image = copy.deepcopy(h_img).astype("float")

for i in range(len(output_image)):
    for j in range(len(output_image[i])):
        output_image[i][j] = numer[i][j]/denom[i][j]
        # print(output_image[i][j],numer[i][j],denom[i][j])
    
output_image += atmospheric_light.astype("float")
output_image[:,:,0] = quantization(output_image[:,:,0],256,[np.min(output_image[:,:,0]),np.max(output_image[:,:,0])])
output_image[:,:,1] = quantization(output_image[:,:,1],256,[np.min(output_image[:,:,1]),np.max(output_image[:,:,1])])
output_image[:,:,2] = quantization(output_image[:,:,2],256,[np.min(output_image[:,:,2]),np.max(output_image[:,:,2])])

print(numer.shape)
print(denom.shape)
print(output_image.shape)

cv2.imwrite("./output/hazy.jpg",h_img)
cv2.imwrite("./output/dehazy.jpg",output_image)