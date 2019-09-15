import cv2
import numpy as np
import argparse

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 15

homography = None

# ORB keypoint detector
orb = cv2.ORB_create()

# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

model = cv2.imread('book.jpeg', 0) # target image

# Compute model keypoints and its descriptors
kp_model, des_model = orb.detectAndCompute(model, None)

# init video capture
cap = cv2.VideoCapture(0)
i = 0
while True:

    ret, frame = cap.read()
    if not ret:
        print ('Unable to capture Video')
        break

    kp_frame, des_frame = orb.detectAndCompute(frame, None)

    matches = bf.match(des_model, des_frame)

    # sort them in the order of their distance
    # the lower the distance, the better the match
    matches = sorted(matches, key=lambda x: x.distance)

    # compute Homography if enough matches are found
    if len(matches) > MIN_MATCHES:
        # differenciate between source points and destination points
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # compute Homography
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = model.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, homography)
        # connect them with lines
        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)


        frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:MIN_MATCHES], 0, flags=2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print ("Not enough matches have been found - %d/%d" % (len(matches),
                                                              MIN_MATCHES))

cap.release()
cv2.destroyAllWindows()
