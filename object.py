import numpy
import cv2

img = cv2.imread('testing.png', 0)

# Initiate ORB Detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img, None)

# compute the descriptions with ORB
kp, des = orb.compute(img, kp)

# draw only keypoint location, not size and orientation
img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=0)
cv2.imshow('keypoints', img2)
cv2.waitKey(0)
