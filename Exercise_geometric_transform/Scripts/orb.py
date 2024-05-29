import numpy as np
import cv2 as cv
from scipy.linalg import solve

img1 = cv.imread('Photos_set_1/horizontal_center.jpg')
img2 = cv.imread('Photos_set_1/horizontal_right.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv.ORB_create()

# kp --> Keypoints : array containing the feature points of the grayscale image
# des --> Descriptor : array of size [nb keypoints]x128 containing informations about feature points

# find the keypoints and compute the descriptors with ORB
kp1 = orb.detect(gray1, None)
kp1, des1 = orb.compute(gray1, kp1)
kp2 = orb.detect(gray2, None)
kp2, des2 = orb.compute(gray2, kp2)

print(len(kp1))
print(kp1[0].pt)

# Draw keypoints
img1_kp = gray1
img1_kp = cv.drawKeypoints(gray1, kp1, img1_kp)
cv.imwrite('orb_keypoints.jpg', img1_kp)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
ratio = 0.7
good_matches = []
for m,n in matches:
    if m.distance < ratio * n.distance:
        good_matches.append([m])

# Draw matches
img_matches = cv.drawMatchesKnn(gray1, kp1, gray2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('orb_matches.jpg', img_matches)

# Construct an array of pair of matched points
matched_points = []
for m in good_matches:
    pt1 = kp1[m[0].queryIdx].pt
    pt2 = kp2[m[0].trainIdx].pt
    matched_points.append([pt1, pt2])


####### Translation model #######

# Jacobian matrix
J = np.array([[1, 0], 
              [0, 1]])
Jt = J.transpose()

A = [[0, 0], 
     [0, 0]]
b = [[0],
     [0]]
for m in matched_points:
    # delta = [[x0 - x1], [y0 - y1]]
    delta = [[m[0][0] - m[1][0]], [m[0][1] - m[1][1]]]
    A = A + Jt.dot(J)
    b = b + Jt.dot(delta)

p = solve(A, b)

print(p)

M = np.array([[1, 0, p[0][0]],
              [0, 1, p[1][0]],
              [0, 0, 0      ]])

###### Stitching ######

img2_translated = cv.warpPerspective(gray2, M, (4640, 2610)) 
cv.imwrite('img2_translated.jpg', img2_translated)
