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
    #cv2.imshow('frame', capt)
    #cv2.waitKey(0)
else:
    print ("Not enough matches have been found - %d/%d" % (len(matches),
                                                          MIN_MATCHES))

# assuming matches stores the matches found and returned by bf.match(des_model, des_frame)
# differentiate between source points and destination points
src_points = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_points = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# compute Homography where 5.0 is the threshold
M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

# Draw a rectangle that marks the found model in the frame
h, w = model.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

# project corners into frame
dst = cv2.perspectiveTransform(pts, M)
print (np.int32(dst))

# connect them with lines
img2 = cv2.polylines(cv2.resize(model, (800, 600)), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
cv2.imshow('frame1', img2)
cv2.waitKey(0)
