import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('./raw_images/img1.jpg', cv.IMREAD_GRAYSCALE)  # referenceImage
img2 = cv.imread('./raw_images/img2.jpg', cv.IMREAD_GRAYSCALE)  # sensedImage

#  Resize the image by a factor of 8 on each side. If your images are
# very high-resolution, you can try to resize even more, but if they are
# already small you should set this to something less agressive.
resize_factor = 1.0 / 8.0

img1_rs = cv.resize(img1, (0, 0), fx=resize_factor, fy=resize_factor)
img2_rs = cv.resize(img2, (0, 0), fx=resize_factor, fy=resize_factor)

# Initiate SIFT detector
sift_detector = cv.SIFT_create()

# Find the keypoints and descriptors with SIFT on the lower resolution images
kp1, des1 = sift_detector.detectAndCompute(img1_rs, None)
kp2, des2 = sift_detector.detectAndCompute(img2_rs, None)

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

# Get low-res and high-res sizes
low_height, low_width = img1_rs.shape
height, width = img1.shape
low_size = np.float32([[0, 0], [0, low_height], [low_width, low_height], [low_width, 0]])
high_size = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

# Compute scaling transformations
scale_up = cv.getPerspectiveTransform(low_size, high_size)
scale_down = cv.getPerspectiveTransform(high_size, low_size)

#  Combine the transformations. Remember that the order of the transformation
# is reversed when doing matrix multiplication
# so this is actualy scale_down -> H -> scale_up
h_and_scale_up = np.matmul(scale_up, H)
scale_down_h_scale_up = np.matmul(h_and_scale_up, scale_down)

# Warp image 1 to align with image 2
img1Reg = cv.warpPerspective(
    img1,
    scale_down_h_scale_up,
    (img2.shape[1], img2.shape[0])
)

# Get the current script's directory
current_dir = os.path.dirname(__file__)

# Construct the path to the output folder
registered_images_folder = os.path.join(current_dir, 'registered_images')

# Assuming img1Reg contains the image data you want to save
# Save the image in the output folder
output_path = os.path.join(registered_images_folder, 'aligned_img1.jpg')
cv.imwrite(output_path, img1Reg)

print(f"图片已存入: {output_path}")