import os
import cv2
import numpy as np

## ORB (Oriented FAST and Rotated BRIEF) feature detector and matcher
## to find correspondences between keypoints in the images

# 计算图像梯度幅值的函数
def calculate_gradient_magnitude(image):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算梯度
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    # 计算平均梯度幅值
    avg_gradient_magnitude = cv2.mean(gradient_magnitude)[0]
    return avg_gradient_magnitude

# Function to perform feature matching between two images using ORB detector
def feature_matching(img1, img2):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors using ORB
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Initialize Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match keypoints
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches based on distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw top matches (for visualization)
    matching_result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Apply affine transformation to align the images
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the affine transformation matrix
    M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts)

    # Apply the transformation to the second image
    rows, cols, _ = img2.shape
    registered_img2 = cv2.warpAffine(img2, M, (cols, rows))

    return registered_img2

# 读取三张图像
img1 = cv2.imread('./raw_images/img1.jpg')
img2 = cv2.imread('./raw_images/img2.jpg')
img3 = cv2.imread('./raw_images/img3.jpg')

# 计算三张图像的梯度幅值
grad_magnitude_img1 = calculate_gradient_magnitude(img1)
grad_magnitude_img2 = calculate_gradient_magnitude(img2)
grad_magnitude_img3 = calculate_gradient_magnitude(img3)

# 找出梯度幅值最大的图像作为基准图像
max_grad_image = None
if grad_magnitude_img1 > grad_magnitude_img2 and grad_magnitude_img1 > grad_magnitude_img3:
    max_grad_image = img1
    print("最清晰的图片是img1")
elif grad_magnitude_img2 > grad_magnitude_img1 and grad_magnitude_img2 > grad_magnitude_img3:
    max_grad_image = img2
    print("最清晰的图片是img2")
else:
    max_grad_image = img3
    print("最清晰的图片是img3")

# # 显示梯度幅值最大的图像
# cv2.imshow('Maximum Gradient Image', max_grad_image)
# cv2.waitKey(0)

# # 关闭所有打开的窗口
# cv2.destroyAllWindows()

# Perform image registration using feature matching between img2 and max_grad_image
# Assuming you have already implemented this function
registered_img2 = feature_matching(img2, max_grad_image)

# Perform image registration using feature matching between img3 and max_grad_image
# Assuming you have already implemented this function
registered_img3 = feature_matching(img3, max_grad_image)

# Get the current script's directory
current_dir = os.path.dirname(__file__)

# Construct the path to the output folder
registered_images_folder = os.path.join(current_dir, 'registered_images')

# Assuming img1Reg contains the image data you want to save
# Save the image in the output folder
output_path2 = os.path.join(registered_images_folder, 'ORBregistered_img2.jpg')
cv2.imwrite(output_path2, registered_img2)
output_path3 = os.path.join(registered_images_folder, 'ORBregistered_img3.jpg')
cv2.imwrite(output_path3, registered_img3)

print(f"图片已存入: {output_path2}")