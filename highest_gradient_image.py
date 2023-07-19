import os
import cv2
import numpy as np

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

# 特征点检测和匹配
def feature_matching(img1, img2):
    # 转换为灰度图像
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 初始化SIFT特征检测器
    sift = cv2.SIFT_create()

    # 检测特征点和计算描述子
    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)

    # 初始化FLANN基于特征匹配器
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 特征点匹配
    matches = flann.knnMatch(des1, des2, k=2)

    # 保留好的匹配结果
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 获取匹配特征点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 对图像进行配准
    aligned_img = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

    return aligned_img

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
output_path2 = os.path.join(registered_images_folder, 'registered_img2.jpg')
cv2.imwrite(output_path2, registered_img2)
output_path3 = os.path.join(registered_images_folder, 'registered_img3.jpg')
cv2.imwrite(output_path3, registered_img3)

print(f"图片已存入: {output_path2}")