import numpy as np
import cv2 as cv
from scipy.linalg import solve

# define useful paths
path = 'Exercise_geometric_transform/'
set1_path = path + 'Photos_set_1/'
set2_path = path + 'Photos_set_2/'
res_path = path + 'Results/'

# read input images and convert it into grey scale images
img1 = cv.imread(set1_path + 'horizontal_center.jpg')
img2 = cv.imread(set1_path + 'horizontal_right.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# initiate ORB detector
orb = cv.ORB_create()

# kp --> Keypoints : array containing the feature points of the grayscale image
# des --> Descriptor : array of size [nb keypoints]x128 containing informations about feature points

# find the keypoints and compute the descriptors with ORB
kp1 = orb.detect(gray1, None)
kp2 = orb.detect(gray2, None)
kp1, des1 = orb.compute(gray1, kp1)
kp2, des2 = orb.compute(gray2, kp2)

print('nb keypoints img1 : ' + len(kp1))

# draw keypoints and save the result image
img1_kp = gray1
img1_kp = cv.drawKeypoints(gray1, kp1, img1_kp)
cv.imwrite(res_path + 'orb_keypoints.jpg', img1_kp)

# match keypoints between both images using brute-force
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# apply ratio test to select only the strongest matches
ratio = 0.7
good_matches = []
for m,n in matches:
    if m.distance < ratio * n.distance:
        good_matches.append([m])

# draw matches and save the result image
img_matches = cv.drawMatchesKnn(gray1, kp1, gray2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite(res_path + 'orb_matches.jpg', img_matches)

# construct an array of pair of matched points
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
cv.imwrite(res_path + 'img2_translated.jpg', img2_translated)
