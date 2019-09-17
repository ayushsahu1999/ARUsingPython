import cv2
import numpy as np


# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 15

homography = None

# ORB keypoint detector
orb = cv2.ORB_create()

# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

model = cv2.imread('book.jpeg', 0) # target image

#replaceImg = cv2.imread('toc.jpg', 0)
#rows, cols = replaceImg.shape
img2 = []

#pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
processing = True
maskThreshold = 10

# Compute model keypoints and its descriptors
kp_model, des_model = orb.detectAndCompute(model, None)

# init video capture
cap = cv2.VideoCapture(0)
vid = cv2.VideoCapture('smffh.mp4')
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

        r, c, ch = frame.shape
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

        while True:
            ret, replaceImg = vid.read()
            rows, cols, ch = replaceImg.shape
            pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])

            # compute the transform matrix
            M1 = cv2.getPerspectiveTransform(pts1, dst)

            # make the perspective change to camera size
            dst1 = cv2.warpPerspective(replaceImg, M1, (c, r))
            #img2 = cv2.addWeighted(img2,1,dst1,1,0)

            # a mask is created for adding two images
            ret, mask = cv2.threshold(dst1, maskThreshold, 1, cv2.THRESH_BINARY_INV)

            # erode and dilate are used to delete the noise
            mask = cv2.erode(mask, (3, 3))
            mask = cv2.dilate(mask, (3, 3))

            print (frame.shape)
            print (dst1.shape)
            print (mask.shape)


            # the two images are added using the mask
            for c in range(0, 3):
                frame[:, :, c] = dst1[:, :, c]*(1 - mask[:, :]) + frame[:, :, c]*mask[:, :]

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print ("Not enough matches have been found - %d/%d" % (len(matches),
                                                              MIN_MATCHES))

cap.release()
vid.release()
cv2.destroyAllWindows()
