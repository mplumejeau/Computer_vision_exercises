import numpy as np
import cv2 as cv
from scipy.linalg import solve

# read 2 input images and convert them into grey scale images
def read_and_grey_images(img1_path, img2_path):
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    return gray1, gray2

# find keypoints of 2 input grey scale images, perform matches between both images and return the strongest matches into an array
def match_keypoints(gray1, gray2, ratio, res_path):

    # initiate ORB detector
    orb = cv.ORB_create()

    # kp --> Keypoints : array containing the feature points of the grayscale image
    # des --> Descriptor : array of size [nb keypoints]x128 containing informations about feature points

    # find the keypoints and compute the descriptors with ORB
    kp1 = orb.detect(gray1, None)
    kp2 = orb.detect(gray2, None)
    kp1, des1 = orb.compute(gray1, kp1)
    kp2, des2 = orb.compute(gray2, kp2)

    print('nb keypoints img1 : ' + str(len(kp1)))

    # draw keypoints and save the result image
    img1_kp = gray1
    img1_kp = cv.drawKeypoints(gray1, kp1, img1_kp)
    cv.imwrite(res_path + 'orb_keypoints.jpg', img1_kp)

    # match keypoints between both images using brute-force
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # apply ratio test to select only the strongest matches
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

    return matched_points

def J_translation(x, y):
    J = np.array([[1, 0], 
                  [0, 1]])
    return J

def J_similarity(x, y):
    J = np.array([[1, 0, x, -y], 
                  [0, 1, y, x]])
    return J

def J_affine(x, y):
    J = np.array([[1, 0, x, y, 0, 0], 
                  [0, 1, 0, 0, x, y]])
    return J

# compute the nxn matrix A and the nx1 vector b from the matched points between 2 images and the Jacobian matrix
def compute_A_b(matched_points, J_def, n):
    
    A = np.zeros([n, n])
    b = np.zeros([n, 1])

    for m in matched_points:

        x1 = m[0][0]
        y1 = m[0][1]
        x2 = m[1][0]
        y2 = m[1][1]
        delta = [[x2 - x1], 
                 [y2 - y1]]

        # Jacobian matrix
        J = J_def(x1, y1)
        Jt = J.transpose()
        
        A = A + Jt.dot(J)
        b = b + Jt.dot(delta)

    return A, b

# compute the translation parameters between 2 images using the matched keypoints and return the 3x3 transformation matrix 
def compute_translation_matrix(matched_points):

    # dimension of matrix A (biggest dim of J)
    n = 2
    
    # compute matrix A and vector b    
    A, b = compute_A_b(matched_points, J_translation, n)

    # compute translation 2x1 vector p 
    p = solve(A, b)
    print('translation vector :')
    print(p)

    # 3x3 transformation matrix
    M = np.array([[1, 0, p[0][0]],
                  [0, 1, p[1][0]],
                  [0, 0, 0      ]])
    
    return M

# compute the euclidean parameters between 2 images using the matched keypoints and return the 3x3 transformation matrix 
def compute_euclidean_matrix(matched_points):

    # for the estimation of the euclidean transformation, we first compute a similarity transformation since it is a non-linear problem

    # dimension of matrix A (biggest dim of J)
    ns = 4
    
    # compute matrix A and vector b    
    As, bs = compute_A_b(matched_points, J_similarity, ns)

    # compute similarity 4x1 vector p 
    ps = solve(As, bs)

    # compute euclidean 3x1 vector p from the similarity vector
    p = np.zeros([3, 1])
    p[0][0] = ps[0][0]
    p[1][0] = ps[1][0]
    p[2][0] = np.arctan(ps[3][0]/ps[2][0])
    print('euclidean vector :')
    print(p)

    # 3x3 transformation matrix
    M = np.array([[np.cos(p[2][0]), -np.sin(p[2][0]), p[0][0]],
                  [np.sin(p[2][0]), np.cos(p[2][0]) , p[1][0]],
                  [0        , 0        , 0      ]])
    
    return M

# compute the similarity parameters between 2 images using the matched keypoints and return the 3x3 transformation matrix 
def compute_similarity_matrix(matched_points):

    # dimension of matrix A (biggest dim of J)
    n = 4
    
    # compute matrix A and vector b    
    A, b = compute_A_b(matched_points, J_similarity, n)

    # compute similarity 4x1 vector p 
    p = solve(A, b)
    print('similarity vector :')
    print(p)

    # 3x3 transformation matrix
    M = np.array([[1+p[2][0], -p[3][0] , p[0][0]],
                  [p[3][0]  , 1+p[2][0], p[1][0]],
                  [0        , 0        , 0      ]])
    
    return M

# compute the affine parameters between 2 images using the matched keypoints and return the 3x3 transformation matrix 
def compute_affine_matrix(matched_points):

    # dimension of matrix A (biggest dim of J)
    n = 6
    
    # compute matrix A and vector b    
    A, b = compute_A_b(matched_points, J_affine, n)

    # compute affine 6x1 vector p 
    p = solve(A, b)
    print('affine vector :')
    print(p)

    # 3x3 transformation matrix
    M = np.array([[1+p[2][0], p[3][0]  , p[0][0]],
                  [p[4][0]  , 1+p[5][0], p[1][0]],
                  [0        , 0        , 0      ]])
    
    return M

# transform the first image using the transformation matrix M and save the result image
# DON T WORK FOR THE MOMENT
def stitch_images(gray1, M, res_path):
    img1_translated = cv.warpPerspective(gray1, M, (4640, 2610)) 
    cv.imwrite(res_path + 'img1_translated.jpg', img1_translated)

####### main #######

# define useful paths
path = 'Exercise_geometric_transform/'
set1_path = path + 'Photos_set_1/'
set2_path = path + 'Photos_set_2/'
res_path = path + 'Results/'

# 2 images with similarities
img1_path = set1_path + 'horizontal_center.jpg'
img2_path = set1_path + 'horizontal_right.jpg'

# ratio applied to selcet the strongest matches
ratio = 0.7

gray1, gray2 = read_and_grey_images(img1_path, img2_path)

matched_points = match_keypoints(gray1, gray2, ratio, res_path)

# compute transformation matrixes to go from img2 to img1 using different transformation
Mt = compute_translation_matrix(matched_points)
Me = compute_euclidean_matrix(matched_points)
Ms = compute_similarity_matrix(matched_points)
Ma = compute_affine_matrix(matched_points)

print('translation matrix :')
print(Mt)
print('euclidean matrix :')
print(Me)
print('similarity matrix :')
print(Ms)
print('affine matrix :')
print(Ma)

stitch_images(gray1, Mt, res_path)
