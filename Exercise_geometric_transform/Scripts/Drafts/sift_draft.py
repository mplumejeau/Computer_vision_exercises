import numpy as np
import cv2 as cv

img1 = cv.imread('Photos_set_1/horizontal_center.jpg')
img2 = cv.imread('Photos_set_1/horizontal_right.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()


# kp --> Keypoints : array containing the feature points of the grayscale image
# des --> Descriptor : array of size [nb keypoints]x128 containing informations about feature points
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(len(kp1))
print(kp1[30000].pt)

img1_kp = cv.drawKeypoints(gray1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('sift_keypoints.jpg', img1_kp)

