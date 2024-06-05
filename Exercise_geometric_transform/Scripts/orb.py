import numpy as np
import cv2 as cv
import random
from scipy.linalg import solve

# read 2 input images and convert them into grey scale images
def read_and_grey_images(img1_path, img2_path):
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    img1 = cv.resize(img1, (0,0), fx=0.2, fy=0.2, interpolation = cv.INTER_AREA)
    img2 = cv.resize(img2, (0,0), fx=0.2, fy=0.2, interpolation = cv.INTER_AREA)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    return img1, img2, gray1, gray2

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

    # extract the 4 best matches from good_matches (those with the smallest distance)
    if len(good_matches) < 4:
        print("less than 4 matches between images")
    else:
        sorted_good_matches = sorted(good_matches, key=lambda x: x[0].distance)
        best_4_matches = sorted_good_matches[:4]

    # draw matches and save the result image
    img_best_matches = cv.drawMatchesKnn(gray1, kp1, gray2, kp2, best_4_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(res_path + 'orb_best_matches.jpg', img_best_matches)

    # construct an array of pair of matched points
    matched_points = []
    for m in good_matches:
        pt1 = kp1[m[0].queryIdx].pt
        pt2 = kp2[m[0].trainIdx].pt
        matched_points.append([pt1, pt2])

    # construct an array of pair of the 4 best matched points
    best_4_matched_points = []
    for m in best_4_matches:
        pt1 = kp1[m[0].queryIdx].pt
        pt2 = kp2[m[0].trainIdx].pt
        best_4_matched_points.append([pt1, pt2])

    return matched_points, best_4_matched_points

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
                  [0, 0, 1      ]])
    
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
                  [0              , 0               , 1      ]])
    
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
                  [0        , 0        , 1      ]])
    
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
                  [0        , 0        , 1      ]])
    
    return M

def get_affine_matrix(matched_points):

    pts1 = []
    pts2 = []

    # creation of a list of 3 points for each image
    for m in matched_points[:3]:
        pts1.append(m[0])
        pts2.append(m[1])

    # conversion into np array
    pts1_np = np.array(pts1, dtype=np.float32)
    pts2_np = np.array(pts2, dtype=np.float32)

    # 3x3 transformation matrix
    M = cv.getAffineTransform(pts1_np, pts2_np)

    return M

def get_homography_matrix(matched_points):

    pts1 = []
    pts2 = []

    # creation of a list of 4 points for each image
    for m in matched_points:
        pts1.append(m[0])
        pts2.append(m[1])

    print(pts1)
    print(pts2)

    # conversion into np array
    pts1_np = np.array(pts1, dtype=np.float32)
    pts2_np = np.array(pts2, dtype=np.float32)

    # 3x3 transformation matrix
    M = cv.getPerspectiveTransform(pts1_np, pts2_np)

    return M

# transform an image using the 2x3 transformation matrix M and save the result image
def transform_affine(gray, M, res_path):

    img_width = gray.shape[1]
    img_height = gray.shape[0]

    img_transformed = cv.warpAffine(gray, M, (img_width, img_height))

    cv.imwrite(res_path + 'img_query_affine_transform.jpg', img_transformed)

# transform an image using the 3x3 transformation matrix M and save the result image
def transform(gray, M, res_path):

    img_width = gray.shape[1]
    img_height = gray.shape[0]

    img_transformed = cv.warpPerspective(gray, M, (img_width, img_height))

    print(img_transformed.shape[:2])

    cv.imwrite(res_path + 'img_query_transform.jpg', img_transformed)

# compute the canvas dimension required to contain the reference and the transformed query images
def calculate_canvas_size(img_query, img_ref, M):
    
    # get images dimensions
    h1, w1 = img_query.shape[:2]
    h2, w2 = img_ref.shape[:2]

    # get corners' positions for both images
    corners_img_query = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners_img_ref = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # get corners' positions of the transformed query image
    transformed_corners_img_query = cv.perspectiveTransform(corners_img_query, M)

    # combine corners of both images
    all_corners = np.vstack((corners_img_ref, transformed_corners_img_query))

    # find limits of the canvas
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = [-xmin, -ymin]

    # canvas dimensions
    canvas_size = (xmax - xmin, ymax - ymin)

    return canvas_size, translation

# stitch the reference image and the transformed query image on a single image
def blend_images(img_query, img_ref, M):

    # compute canva dimension
    canvas_size, translation = calculate_canvas_size(img_query, img_ref, M)
    tx, ty = translation

    # multiplication of transformation matrix and translation matrix
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    M_translation = np.dot(translation_matrix, M)

    # canvas creation
    canvas = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)

    # put ref image on the canvas
    canvas[ty:ty+img_ref.shape[0], tx:tx+img_ref.shape[1]] = img_ref

    # transformation of the query image 
    img_query_transformed = cv.warpPerspective(img_query, M_translation, canvas_size)
    
    # blend images on the canvas
    canvas = cv.addWeighted(canvas, 1, img_query_transformed, 1, 0)

    return canvas



####### main #######

# define useful paths
path = 'Exercise_geometric_transform/'
set1_path = path + 'Photos_set_1/'
set2_path = path + 'Photos_set_2/'
set3_path = path + 'Photos_set_3/'
res_path = path + 'Results/'

# 2 images with similarities
img_query_path = set1_path + 'vertical_rotate.jpg'  # the image i want to deform in order to obtain the ref image
img_ref_path = set1_path + 'vertical_center.jpg'    # the reference image

# ratio applied to select the strongest matches
ratio = 0.7

# read and resize images
img_query, img_ref, gray_query, gray_ref = read_and_grey_images(img_query_path, img_ref_path)

print(gray_query.shape[:2])
print(gray_ref.shape[:2])

# find keypoints in both images and matched them
matched_points, best_4_matched_points = match_keypoints(gray_query, gray_ref, ratio, res_path)

# compute transformation matrixes to go from img_query to img_ref using different transformations
Mt = compute_translation_matrix(matched_points)
Me = compute_euclidean_matrix(matched_points)
Ms = compute_similarity_matrix(matched_points)
Ma = compute_affine_matrix(matched_points)

Ma2 = get_affine_matrix(best_4_matched_points)
Mh = get_homography_matrix(best_4_matched_points)

# print transformation matrixes of different transformations
print('translation matrix :')
print(Mt)
print('euclidean matrix :')
print(Me)
print('similarity matrix :')
print(Ms)
print('affine matrix :')
print(Ma)

print('affine matrix from func :')
print(Ma2)
print('homography matrix from func :')
print(Mh)

# transform the query image into the ref image
transform(gray_query, Ma, res_path)

# blend query and ref images on a canvas
canvas = blend_images(gray_query, gray_ref, Ma)
cv.imwrite(res_path + 'stitched_images.jpg', canvas)
