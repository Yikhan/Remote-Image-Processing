# -*- coding: utf-8 -*-
import sys
import cv2
import math
from matplotlib import pyplot as plt

AREA_thres = 0.001
EPSILON_thres = 0.002
COLOR_DIS_thres = 140

# file path, not used
Sample = "Sample"
Sample_Pre = "Sample_Pre"

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
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(imgray, (5,5), 0)
    threshed_img = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    return threshed_img

def simple_threshold(pic_name, pic_origin_name=None, mode=0):
    
    img = cv2.imread(pic_name)
    # convert to binary
    if mode == 0:
        img_thresholded = binary(img)
    elif mode == 1:
        img_thresholded = adaptive_binary(img)
    elif mode == 2:
        img_thresholded = otsu_binary(img)
 
    if pic_origin_name is not None:
        img_origin = cv2.imread(pic_origin_name)
    else:
        img_origin = img.copy()
    
    # morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    img_thresholded = cv2.erode(img_thresholded, kernel)
    # contour check
    img_contour = img_origin.copy()
    contour_find(img_thresholded, img_contour)

    cv2.imshow("Origin", img_origin)
    cv2.imshow("Contour", img_contour)
    cv2.imshow("Binary", img_thresholded)
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
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
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
        
        cv2.drawContours(img_origin, [approx], 0, (0,255,0), 3)
        print("=== valid contour for {} ===".format(valid_shape_index))
        valid_shape_index += 1
        print("Area: ", area)
        for v in approx:
            print(v)

# end of function
    
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