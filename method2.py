import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('./raw_images/img1.jpg', cv.IMREAD_GRAYSCALE)  # referenceImage
img2 = cv.imread('./raw_images/img2.jpg', cv.IMREAD_GRAYSCALE)  # sensedImage

#img1 = cv.imread('img1.jpg')  # referenceImage
#img2 = cv.imread('img2.jpg', cv.IMREAD_GRAYSCALE)  # sensedImage

# Initiate SIFT detector
sift_detector = cv.SIFT_create()
# Find the keypoints and descriptors with SIFT
kp1, des1 = sift_detector.detectAndCompute(img1, None)
kp2, des2 = sift_detector.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Filter out poor matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

matches = good_matches

points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

# Find homography
H, mask = cv.findHomography(points1, points2, cv.RANSAC)

# Warp image 1 to align with image 2
img1Reg = cv.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

# Get the current script's directory
current_dir = os.path.dirname(__file__)

# Construct the path to the output folder
registered_images_folder = os.path.join(current_dir, 'registered_images')

# Assuming img1Reg contains the image data you want to save
# Save the image in the output folder
output_path = os.path.join(registered_images_folder, 'aligned_img2.jpg')
cv.imwrite(output_path, img1Reg)

print(f"图片已存入: {output_path}")
