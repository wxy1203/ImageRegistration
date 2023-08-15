import os
import cv2
import numpy as np


# 计算图像梯度幅值的函数
def calculate_average_gradient_magnitude(image):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算梯度
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    # 计算平均梯度幅值
    avg_grad = cv2.mean(grad)[0]
    return avg_grad


# 特征点检测和匹配 SIFT算法传入参数
def feature_matching(src, dst, sift_ratio=0.7):
    # Convert to grayscale
    gray_img_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_img_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT feature detector
    sift = cv2.SIFT_create()

    # Detect keypoint and compute descriptors
    kp_src, des_src = sift.detectAndCompute(gray_img_src, None)
    kp_dst, des_dst = sift.detectAndCompute(gray_img_dst, None)

    # Initialize FLANN-based feature matcher
    flann_index_kdtree = 0
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Feature matching
    matches = flann.knnMatch(des_src, des_dst, k=2)

    # Select good matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < sift_ratio * n.distance:
            good_matches.append(m)

    # Extract coordinates of matching keypoint
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Perform image alignment
    aligned_img = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))

    return aligned_img


def image_reg(reg_img_dir_path, raw_img_dir_path):

    # Get a list of image files in the specified directory
    lst_img_files = [f for f in os.listdir(raw_img_dir_path) if f.endswith('.jpg') or f.endswith('.png')]

    # Get creation timestamps for each image 获得创建时间
    img_timestamps = [os.path.getctime(os.path.join(raw_img_dir_path, f)) for f in lst_img_files]

    # Sort images based on their timestamps 按照创建顺序 由最早到最近
    sorted_img_files = [f for _, f in sorted(zip(img_timestamps, lst_img_files))]

    # Read images and create a list of image objects
    lst_images = [cv2.imread(os.path.join(raw_img_dir_path, f)) for f in sorted_img_files]

    # 找到最清晰的照片，作为基准照片
    max_grad = calculate_average_gradient_magnitude(lst_images[0])
    std_img = lst_images[0]
    max_grad_img_idx = 1
    for i in range(len(lst_images)-1):
        grad = calculate_average_gradient_magnitude(lst_images[i+1])
        if grad > max_grad:
            max_grad = grad
            std_img = lst_images[i+1]
            max_grad_img_idx = i+1
     
    print("最清晰的图片是img" + str(max_grad_img_idx))

    # # 显示梯度幅值最大的图像
    # cv2.imshow('Maximum Gradient Image', std_img)
    # cv2.waitKey(0)

    # # 关闭所有打开的窗口
    # cv2.destroyAllWindows()

    for i, img in enumerate(lst_images):
        reg_img = feature_matching(img, std_img)

        # Save registered image
        file_output_path = os.path.join(reg_img_dir_path, f"registered_{i + 1}.jpg")
        cv2.imwrite(file_output_path, reg_img)

        print(f"图片已存入: {file_output_path}")

    # registered_img2 = feature_matching(LstImg[1], std_img)
    #
    # # Perform image registration using feature matching between img3 and std_img
    # # Assuming you have already implemented this function
    # registered_img3 = feature_matching(LstImg[2], std_img)
    #
    # registered_img4 = feature_matching(img4, std_img)
    # registered_img5 = feature_matching(img5, std_img)
    #
    #
    #
    #
    # output_path3 = os.path.join(registered_images_folder, 'registered_img3.jpg')
    # cv2.imwrite(output_path3, registered_img3)
    #
    # output_path4 = os.path.join(registered_images_folder, 'registered_img4.jpg')
    # cv2.imwrite(output_path4, registered_img4)
    # output_path5 = os.path.join(registered_images_folder, 'registered_img5.jpg')
    # cv2.imwrite(output_path5, registered_img5)


