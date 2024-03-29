import numpy as np
import cv2

model = cv2.imread('book.jpeg', 0) # target image
img = cv2.imread('book_in_frame.jpeg', 0) # query image

# Initiate ORB Detector
orb = cv2.ORB_create()

MIN_MATCHES = 15

# ORB keypoint detector
orb = cv2.ORB_create()

# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Compute model keypoints and its descriptors
kp_model, des_model = orb.detectAndCompute(model, None)

# Compute scene keypoints and its descriptors
kp_frame, des_frame = orb.detectAndCompute(img, None)

# Match frame descriptors with model descriptors
matches = bf.match(des_model, des_frame)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

if len(matches) > MIN_MATCHES:
    # draw first 15 matches.
    capt = cv2.drawMatches(model, kp_model, img, kp_frame,
                      matches[:MIN_MATCHES], 0, flags=2)
    # show result
    #cv2.imshow('frame', cv2.resize(capt, (800, 600)))
    #cv2.waitKey(0)
else:
    print ("Not enough matches have been found - %d/%d" % (len(matches),
                                                          MIN_MATCHES))

# assuming matches stores the matches found and
# returned by bf.match(des_model, des_frame)

# differenciate between source points and destination points
src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# compute Homography
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Draw a rectangle that marks the found model in the frame
h, w = model.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
# project corners into frame
dst = cv2.perspectiveTransform(pts, M)
# connect them with lines
img2 = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
cv2.imshow('frame', cv2.resize(img2, (800, 600)))
cv2.waitKey(0)
