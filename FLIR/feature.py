import os
import cv2
import time
import numpy as np

img1 = cv2.imread('ir_record/ir_17_46.png', 0)
img2 = cv2.imread('ir_record/ir_17_54.png', 0)
print(img1.shape, img2.shape)

orb = cv2.ORB_create(nfeatures = 2000)
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
start = time.time()
for i in range(10):
    kp1, des1 = orb.detectAndCompute(img1,None)
print((time.time() - start)/10 * 100)

# fast = cv2.FastFeatureDetector_create()
# brisk = cv2.BRISK_create()
# kp1, des1 = brisk.compute(img1, fast.detect(img1, None)) 
# kp2, des2 = brisk.compute(img2, fast.detect(img2, None)) 
# start = time.time()
# for i in range(10):
#     kp1, des1 = brisk.compute(img1, fast.detect(img1, None)) 
# print((time.time() - start)/10 * 100)

print('detect', len(kp1), len(kp2))

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
print(len(matches))
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:500], None, flags=2)

cv2.imwrite('feature_match.jpg', img3)
cv2.imshow('result', img3)
cv2.waitKey(0)