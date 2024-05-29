import numpy as np
import cv2 as cv
import glob
import os
 
# number of corners
width = 7
height = 6

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((height*width,3), np.float32)
objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
# Clear images_calibrated directory
calibrated_dir = 'images_calibrated'
for file_name in os.listdir(calibrated_dir):
    file_path = os.path.join(calibrated_dir, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)

images = glob.glob('images/*.png')

nb_success = 0

for fname in images:
 
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (width,height), None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
    
        # Draw and display the corners
        cv.drawChessboardCorners(img, (width,height), corners2, ret)
        resized_img = cv.resize(img, (928, 522))
        cv.imshow('img', resized_img)
        cv.waitKey(500)

        # Save the calibrated image
        filename = os.path.join('images_calibrated', os.path.basename(fname))
        cv.imwrite(filename, img)

        nb_success += 1
 
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

rmat = []
for rvec in rvecs:
    R, _ = cv.Rodrigues(rvec)
    rmat.append(R)

print("nb image successfully calibrated :")
print(nb_success)
print()
print("ret :")
print(ret)
print()
print("distortion parameters :")
print(dist)
print()
print("intrinsic parameters :")
print(mtx)
print()
print("extrinsic parameters :")
for i in range(nb_success):
    print()
    print("image ", i)
    print(rmat[i])
    print(tvecs[i])