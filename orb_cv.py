# Read image 
img2 = cv2.imread('./test_data/'+'KAIST_64'+'.jpeg')
orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=4045)
#orb = cv2.ORB_create()
start = tm.time()
kp2 = orb.detect(img2)
stop = tm.time()
executionTimeFinal = (stop - start)
print("The Feature Detection Final Execution Time = ", executionTimeFinal, " Seconds")  
   

# no.of keypoints 
print(len(kp2))
print('Feature points are detected')
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), \
        flags=cv2.DrawMatchesFlags_DEFAULT)

#plt.figure()
plt.figure(figsize=(10,7.5))
plt.imshow(img2_kp)
plt.show()