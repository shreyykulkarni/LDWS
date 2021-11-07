# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:50:49 2021

@author: Atharv Kulkarni
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os


def grayscale(img):
    A = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return A


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    '''The idea is to fill the entire image other than the ROI with the colour black'''
    
    # Defining a blank mask to start with
    mask = np.zeros_like(img)

    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    #plt.figure(dpi=400)
    plt.imshow(masked_image)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    This funciton takes the lines and coordinates from the ROI, takes
    a slope and draws lines in accordance to them.
    This function draws `lines` with `color` and `thickness`.
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    input for this function should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    The output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result weighted image is computed as follows:
    initial_img * α + img * β + λ
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_frame(image):
    global first_frame

    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # hsv = [hue, saturation, value]

    lower_yellow = np.array([20, 50, 100], dtype = "uint8")
    upper_yellow = np.array([30, 200, 255], dtype="uint8")
    
    
    # Yellow and white masks used to accentuate the lane lines for the canny edge detector
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 230, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)
    
    
    # Gaussian blur
    kernel_size = 1
    gauss_gray = gaussian_blur(mask_yw_image,kernel_size)
    
    
    # Thresold values for canny
    low_threshold = 50
    high_threshold = 150
    canny_edges = canny(gauss_gray,low_threshold,high_threshold)

    imshape = image.shape
    
    
    '''AUTOMATED ROI'''
    # lower_left = [imshape[1]/9,imshape[0]]
    # lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
    # top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
    # top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
    
    '''MANUAL ROI'''
    lower_left = [33,225]
    lower_right = [266,225]
    top_left = [135,100]
    top_right = [220,100]


    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    roi_image = region_of_interest(canny_edges, vertices)


    #rho and theta are the distance and angular resolution of the grid in Hough space
    rho = 1
    theta = np.pi/180
    
    #threshold is minimum number of intersections in a grid for the hough lines of the gradient pixels
    threshold = 20
    min_line_len = 50
    max_line_gap = 200

    line_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
    plt.figure(dpi=400)
    plt.imshow(line_image)
    result = weighted_img(line_image, image, α=0.8, β=1.5, λ=0.)
    plt.figure(dpi=400)
    plt.imshow(result)
    return result

'''DISPLAYING THE IMAGE'''

plt.figure(dpi=400)
# Enter image path below
path = 'test_images\image_6.jpg'
plt.imshow(plt.imread(path))
image = cv2.imread(path)

height, width = image.shape[:2]
print('Dimensions are:',height,'*', width)
processed = process_frame(image)
processed = cv2.resize(processed, (500,500))
# cv2.imshow("result",processed)
# cv2.waitKey(0)
