# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 21:01:39 2021

@author: Atharv Kulkarni
"""
import ctypes
import cv2
import numpy as np
from main import LDI
from main import LDV
from matplotlib import pyplot as plt


plt.imshow(plt.imread('image_6.jpg'))
plt.show()

img_s = cv2.imread('image_6.jpg',0)
height, width = img_s.shape[:2]
print(height, width)


# Lane_side=input("Enter which lane you want to detect (Right/Left): \n")
# print("WARNING: Please select coordiantes in accordance to which lane you want to depart to")
# L1=input("Enter L1:")
# L2=input("Enter L2:")
# H1=input("Enter H1:")
# H2=input("Enter H2:")


# if Lane_side=="Right":
#     ctypes.windll.user32.MessageBoxW(0, "WARNINIG!! Vehicle departing towards the Right lane", "ALERT", 1)
# elif Lane_side=="Left":
#     ctypes.windll.user32.MessageBoxW(0, "WARNINIG!! Vehicle departing towards the Left lane", "ALERT", 1)


immage = 'image_6.jpg'
A=LDI(immage,2300,5180,2000,2500)
# cap = cv2.VideoCapture("test2.mp4")
# B = LDV(immage,cap,L1,L2,H1,H2)




