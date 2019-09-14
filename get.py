import numpy as np
import cv2

img = cv2.imread('testing.png', 0)
model = cv2.imread('model.jpeg', 0)

# Initiate ORB Detector
orb = cv2.ORB_create()

MIN_MATCHES = 15

# create bruteforce matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# compute model keywords and its descriptors
kp_model, des_model = orb.detectAndCompute(cv2.resize(model, (800, 600)), None)

# compute scene keywords and its descriptors
kp_frame, des_frame = orb.detectAndCompute(cv2.resize(img, (800, 600)), None)

# match frame descriptors with model descriptors
matches = bf.match(des_model, des_frame)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

if len(matches) > MIN_MATCHES:
    # draw first 15 matches.
    capt = cv2.drawMatches(cv2.resize(model, (800, 600)), kp_model, cv2.resize(img, (800, 600)), kp_frame,
                          matches[:MIN_MATCHES], 0, flags=2)
    # show result
    cv2.imshow('frame', capt)
    cv2.waitKey(0)
else:
    print ("Not enough matches have been found - %d/%d" % (len(matches),
                                                          MIN_MATCHES))
