import numpy as np
import cv2
import pysift
from matplotlib import pyplot as plt
import logging

logger = logging.getLogger(__name__)

import time as tm

MIN_MATCH_COUNT = 10
DIRPATH = "/Users/bilaldastagir/Documents/vscode/FEDHA/"

start = tm.time()
print("Program Started Successfully !!!")
img1 = cv2.imread(DIRPATH + 'images/KAIST.jpeg')  # queryImage
img2 = cv2.imread(DIRPATH + 'images/KAIST2.jpeg')  # trainImage
print(img1.size)

lap1 = tm.time()
executionTime = (lap1 - start)
print("Images are read - The Execution Time = ", executionTime, " Seconds")

# Compute SIFT keypoints and descriptors
#using userdefined function
#kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)
#kp2, des2 = pysift.computeKeypointsAndDescriptors(img2)

#*************************************************
#using built in cv2 library
#sift = cv2.SIFT_create()
#kp1, des1 = sift.detectAndCompute(img1,None)
#kp2, des2 = sift.detectAndCompute(img2,None)
#*************************************************

# Initiate ORB detector
orb = cv2.ORB_create()

# Convering to Gray
img1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

lap2 = tm.time()
executionTime = (lap2 - lap1)
print("Computed SIFT keypoints and descriptors - The Execution Time = ", executionTime, " Seconds")


# no.of keypoints
print(len(des1))
print(len(des2))

# Initialize and use FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
#flann = cv2.FlannBasedMatcher(index_params, search_params)
#matches = flann.knnMatch(des1, des2, k=2)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

lap3 = tm.time()
executionTime = (lap3 - lap2)
print("Initialized and used FLANN - The Execution Time = ", executionTime, " Seconds")

# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

lap4 = tm.time()
executionTime = (lap4 - lap3)
print("Performed Lowe's ratio test - The Execution Time = ", executionTime, " Seconds")

if len(good) > MIN_MATCH_COUNT:
    ilap = tm.time()
    # Estimate homography between template and scene
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    lap5 = tm.time()
    executionTime = (lap5 - ilap)
    print("Estimated homography between template and scene - The Execution Time = ", executionTime, " Seconds")

    # Draw detected template in scene image
    h, w = img1.shape
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2

    lap6 = tm.time()
    executionTime = (lap6 - lap5)
    print("Drawn detected template in scene image - The Execution Time = ", executionTime, " Seconds")

    # Draw SIFT keypoint matches
    for m in good:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))

    lap7 = tm.time()
    executionTime = (lap7 - lap6)
    print("Drawn SIFT keypoint matches - The Execution Time = ", executionTime, " Seconds")

    plt.imshow(newimg)
    plt.show()
    stopLoop = tm.time()
    executionTimeLoop = (stopLoop - start)
    print("The SIFT Loop Execution Time = ", executionTimeLoop, " Seconds")

else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    stop = tm.time()
    executionTimeFinal = (stop - start)
    print("ELSE ----- The SIFT Final Execution Time = ", executionTimeFinal, " Seconds")

stop = tm.time()
executionTimeFinal = (stop - start)
print("The SIFT Final Execution Time = ", executionTimeFinal, " Seconds")

print("Program Ended Successfully !!!")