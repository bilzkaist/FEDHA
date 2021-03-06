import numpy as np
import cv2
from matplotlib import pyplot as plt

DIRPATH = "/Users/bilaldastagir/Documents/vscode/FEDHA/"

# Read image 
img1 = cv2.imread(DIRPATH + 'images/KAIST.jpeg')
img2 = cv2.imread(DIRPATH + 'images/KAIST2.jpeg')

# Initiate ORB detector
orb = cv2.ORB_create()

# Convering to Gray
img1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# no.of keypoints 
print(len(des1))
print(len(des2))

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Create images with keypoints
img1=cv2.drawKeypoints(img1,kp1,img1)
cv2.imwrite('2orb_keypoints.jpg',img1)

img2=cv2.drawKeypoints(img2,kp2,img2)
cv2.imwrite('3orb_keypoints.jpg',img2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print("matches ")
print(len(good))

