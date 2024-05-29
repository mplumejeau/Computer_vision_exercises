import cv2
import numpy as np

def detect_and_match_features(image1, image2, ratio_thresh=0.75):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize the AKAZE detector
    akaze = cv2.AKAZE_create()

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = akaze.detectAndCompute(gray1, None)
    print(descriptors1)
    keypoints2, descriptors2 = akaze.detectAndCompute(gray2, None)

    # Use BFMatcher to find matches between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append([m])
            #print([m])

    print("")

    # Draw match results between both images
    image_match = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, good_matches[1:len(good_matches)], None, flags=2)
    cv2.imwrite('matches.jpg', image_match)

    return keypoints1, keypoints2, good_matches

def find_homography(keypoints1, keypoints2, matches):
    # Extract location of good matches
    if len(matches) > 4:
        src_pts = np.float32([keypoints1[m[0].queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m[0].trainIdx].pt for m in matches]).reshape(-1, 2)

        # Compute homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
    else:
        H, matches_mask = None, None

    return H, matches_mask

def stitch_images_with_transparency(image1, image2, H, alpha=0.5):
    # Get dimensions of the images
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Get the canvas dimensions
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Transform points from the second image
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)

    # Get the bounding box of the resulting panorama
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

    # Translation matrix to shift the panorama
    translation_dist = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Warp the second image
    warped_image1 = cv2.warpPerspective(image1, H_translation.dot(H), (xmax - xmin, ymax - ymin))
    warped_image2 = np.zeros_like(warped_image1)
    warped_image2[translation_dist[1]:h1 + translation_dist[1], translation_dist[0]:w1 + translation_dist[0]] = image2

    # Create a mask to identify the regions where images overlap
    mask1 = np.where(warped_image2.sum(axis=-1) > 0, 1, 0).astype(np.float32)
    mask2 = np.where(warped_image1.sum(axis=-1) > 0, 1, 0).astype(np.float32)

    # Alpha blend the images
    result_img = ((1 - alpha) * warped_image2 + alpha * warped_image1).astype(np.uint8)

    return result_img


# Load images
image1 = cv2.imread('Photos_set_2/center.jpg')
image2 = cv2.imread('Photos_set_2/rotate.jpg')

# Step 1: Detect and match features
keypoints1, keypoints2, good_matches = detect_and_match_features(image1, image2)

# Step 2: Find homography
H, matches_mask = find_homography(keypoints1, keypoints2, good_matches)

if H is not None:

    # Step 3: Stitch images with transparency
    result_img = stitch_images_with_transparency(image1, image2, H, alpha=0.5)

    # Save and display the result
    cv2.imwrite('panorama.jpg', result_img)
    cv2.imshow('Panorama', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough matches found to compute homography.")
