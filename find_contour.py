# -*- coding: utf-8 -*-
import sys
import cv2
import math
from matplotlib import pyplot as plt

import filter as fi

AREA_thres = 0.003
EPSILON_thres = 0.002
COLOR_DIS_thres = 140

# file path
Sample_Pre = "Sample-Pre/"
Sample_Ori = "Sample/"

def test_img(pic_name):
    
    img = cv2.imread(Sample_Ori + pic_name)
    b,g,r = cv2.split(img)
    
    gray_img = fi.adjust_spetrum_BGR(img)
    
    cv2.imshow("Origin", img)
    cv2.imshow("Gray", gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def simple_threshold(pic_name, mode=0):
    
#    img = cv2.imread(Sample_Pre + pic_name)
    img_origin = cv2.imread(Sample_Ori + pic_name)
    img = fi.adjust_spetrum_BGR(img_origin)
    
    # convert to binary
    if mode == 0:
        img_thresholded = fi.binary(img)
    elif mode == 1:
        img_thresholded = fi.adaptive_binary(img)
    elif mode == 2:
        img_thresholded = fi.otsu_binary(img)
 
    # morphology
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    
    img_thresholded = cv2.erode(img_thresholded, kernel_erode)
    img_thresholded = cv2.dilate(img_thresholded, kernel_dilate)
    
    # median filter to make area smoothed
    img_thresholded = cv2.medianBlur(img_thresholded, 9)
     
    # contour check
    img_contour = img_origin.copy()
    contour_find(img_thresholded, img_contour)

    cv2.imshow("Origin", img_origin)
    cv2.imshow("Contour", img_contour)
    cv2.imshow("Binary", img_thresholded)
    cv2.imshow("Gray", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#    images = [img_origin, img, img_thresholded, img_contour]
#    titles = ["Original", "Filtered", "Binary", "Contour"]
#  
#    for i  in range(4):
#        plt.subplot(2,2,i+1)
#        plt.imshow(images[i])
#        plt.title(titles[i])
#        plt.xticks([])
#        plt.yticks([])
#    plt.show()

def contour_find(img, img_origin):
 
    im2, contours, hierarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # draw each contour
    valid_shape_index = 1
    
    for i in range(0, len(contours)):
        cnt = contours[i]
        # invalid area with less than 3 vertex
        if len(cnt) < 3:
            continue
        # invalid area which is too small
        area = cv2.contourArea(cnt) 
        if area < img.shape[0] * img.shape[1] * AREA_thres:
            continue
            
        epsilon = EPSILON_thres * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        cv2.drawContours(img_origin, [approx], 0, (255,0,0), 3)
        
        print("=== valid contour for {} ===".format(valid_shape_index))
        valid_shape_index += 1
        print("Area: ", area)
        for v in approx:
            print(v)
    
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
    img_origin = img.copy()
    
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
     
    contour_find(img_eroded, img_origin)
    
    cv2.imshow("Clustered", img_eroded)
    cv2.imshow("Dis", img)
    cv2.imshow("Origin", img_origin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()