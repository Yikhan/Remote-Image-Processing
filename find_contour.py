# -*- coding: utf-8 -*-
"""
@author: Yikhan
Remote Image Processing
Finding the construction/sand area

----------------------------------------------------------
updated 2018/02/08
"""

import cv2
import timeit
import os
import json
#from matplotlib import pyplot as plt

import filter as fi

AREA_thres = 0.003
EPSILON_thres = 0.002

# file path for reading in images
Sample_Ori = "Sample/"
Sample_Output = "Sample_Output/"

# Test funtion 
def test_img(pic_name):
    
    img = cv2.imread(Sample_Ori + pic_name)
    
    start = timeit.default_timer()
    gray_img = fi.adjust_spetrum_BGR(img)
    time_cost = timeit.default_timer() - start
    
    print("Time cost: ", time_cost)
    
    cv2.imshow("Origin", img)
    cv2.imshow("Gray", gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Batch process from directory
def batch_process():
    
    def get_resolution(gps_pos, img_shape):
        '''
        Depending on the format of GPS data, longitude and latitude
        which might come first in gps_pos
        Here we assume latitude comes first, follwed by longitude
        '''
        gps_start_x, gps_start_y, gps_end_x, gps_end_y = \
         gps_pos["start"][1], gps_pos["start"][0], gps_pos["end"][1], gps_pos["end"][0]
            
        img_width, img_height = img_shape[1], img_shape[0]
        
        x_reso = (abs(gps_end_x) - abs(gps_start_x)) / img_width
        y_reso = (abs(gps_end_y) - abs(gps_start_y)) / img_height
        
        return {"start_x":gps_start_x, "start_y":gps_start_y, "reso_x":x_reso, "reso_y":y_reso}
    
    def convert_contour_toGPS(contour, measure):
        '''
        Convert coordinates of contour from pixel to gps
        '''
        converted_contour = []
        for pos in contour:
            longitude = measure["start_x"] + pos[0][0]*measure["reso_x"]
            latitude = measure["start_y"] - pos[0][1]*measure["reso_y"]
            converted_contour.append([latitude, longitude])
            
        return converted_contour
    
    images = os.listdir(Sample_Ori)
    image_dict = load_json("sample.json")
    
    for img_name in image_dict:
        img = cv2.imread(Sample_Ori + img_name)
        measure = get_resolution(image_dict[img_name], img.shape)
        print(measure)
        contour_list = simple_threshold(img, run_mode=2)
        for i in range(len(contour_list)):
            contour_list[i] = convert_contour_toGPS(contour_list[i], measure)
            print(contour_list[i])
#        print(contour_result)

        
# Read json file
def load_json(json_name):
    
    with open(Sample_Ori + json_name, "r") as load_f:
        load_dict = json.load(load_f)
       
    return load_dict

def simple_threshold(img_origin, binary_mode=0, run_mode=0):
    '''
     Main function for processing the image
     running mode 
     0 if need to show result
     1 if need to save processed image
     2 otherwise
    '''

    img = fi.adjust_spetrum_BGR(img_origin)
    
    # convert to binary
    if binary_mode == 0:
        img_thresholded = fi.binary(img)
    elif binary_mode == 1:
        img_thresholded = fi.adaptive_binary(img)
    elif binary_mode == 2:
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
    contour_result = contour_find(img_thresholded, img_contour)
    
    if run_mode == 0:
        # Show results in windows
        cv2.imshow("Origin", img_origin)
        cv2.imshow("Contour", img_contour)
        cv2.imshow("Binary", img_thresholded)
        cv2.imshow("Gray", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif run_mode == 1:
        # Save images
        cv2.imwrite(Sample_Output+pic_name, img_contour)
        
# -------------- Show images with Pyplot --------------------    
#    images = [img_origin, img, img_thresholded, img_contour]
#    titles = ["Original", "Filtered", "Binary", "Contour"]
#    for i  in range(4):
#        plt.subplot(2,2,i+1)
#        plt.imshow(images[i])
#        plt.title(titles[i])
#        plt.xticks([])
#        plt.yticks([])
#    plt.show()
# -----------------------------------------------------------
        
    return contour_result
    
def contour_find(img, img_origin):
    '''
    Find the contours on a given binary image
    '''
 
    im2, contours, hierarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # draw each contour
    valid_shape_index = 1
    # return list 
    contour_result = []
    
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
        
        contour_result.append(approx)
        
    return contour_result
    
