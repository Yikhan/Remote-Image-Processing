# -*- coding: utf-8 -*-
"""
@author: Yikhan
"""
import sys
import cv2
import math
import random as rd
import numpy as np
from sklearn.linear_model import LinearRegression

def binary(img):
    # convert to gray scale
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(imgray, (9,9), 0)
    ret, threshed_img = cv2.threshold(blur, 200, 255, 0)
    # invert image
    return threshed_img

def otsu_binary(img):
    # convert to gray scale
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(imgray, (5,5), 0)
    ret, threshed_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshed_img

def adaptive_binary(img):
    # convert to gray scale
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(imgray, (5,5), 0)
    threshed_img = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    return threshed_img

# --------------------------------------------
# Parameters solven from images filtered by Photoshop
def sample_extract(img_source_name, img_output_name, sample_number):
    
    img_source = cv2.imread(img_source_name)
    img_output = cv2.imread(img_output_name)
    
    height_max = img_source.shape[0]
    width_max = img_source.shape[1]
    
    sample_x = []
    sample_y = []
    
    for i in range(sample_number):
        pos = (rd.randint(0,height_max-1), rd.randint(0,width_max-1))
        # 3 channels as x from original image
        sample_x.append(img_source[pos[0], pos[1]])
        # 1 channel as y from binary image
        sample_y.append(img_output[pos[0], pos[1]])
        
    return sample_x, sample_y

def sample_regression(x, y):
    
    Model = LinearRegression()
    Model.fit(x,y)
    
    return Model

# --------------------------------------------

def adjust_spetrum_BGR(img):
    
    gray_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            
            pixel = 1.32 * int(-1.522*img[i,j][0] - 1.268*img[i,j][1] + 3.568*img[i,j][2] + 20.810) - 40
            
            if pixel < 0:  pixel = 0
            elif pixel > 255: pixel = 255
            
            gray_img[i,j] = (pixel, pixel, pixel)
            
    return gray_img



COLOR_DIS_thres = 140

def color_distance_BGRWeigh(pixel_1, pixel_2):
    
    r_bar = (pixel_1[2] + pixel_2[2]) / 2
    delta_R = pixel_1[2] - pixel_2[2]
    delta_G = pixel_1[1] - pixel_2[1]
    delta_B = pixel_1[0] - pixel_2[0]
    
    delta_C = math.sqrt(
            (2 + r_bar/256) * delta_R**2 +
            (4 * delta_G**2) +
            (2 + (255-r_bar)/256) * delta_B**2
    )
    
    return delta_C

def color_distance_BGR(pixel_1, pixel_2):
    
    delta_C = math.sqrt(
           (pixel_1[2]-pixel_2[2])**2 +
           (pixel_1[1]-pixel_2[1])**2 +
           (pixel_1[0]-pixel_2[0])**2       
    )
    
    return delta_C 

def color_cluster(pic_name):
    
    img = cv2.imread(pic_name)
    
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if color_distance_BGR(img[i,j], (60, 255, 255)) < COLOR_DIS_thres:
                # set pixel to black
                img[i,j] = (0, 0, 0)
            else:
                img[i,j] = (255, 255, 255)
                
    # convert to gray scale
    img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    img_dialted = cv2.dilate(img, kernel)  
    img_eroded = cv2.erode(img_dialted, kernel)
    
    return img_eroded
            
            
    
            
    
    
    
