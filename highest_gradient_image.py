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

# 特征点检测和匹配
def feature_matching(src, dst):
    # 转换为灰度图像
    gray_img_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_img_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # 初始化SIFT特征检测器
    sift = cv2.SIFT_create()

    # 检测特征点和计算描述子 keypoint & descriptor
    kp_src, des_src = sift.detectAndCompute(gray_img_src, None)
    kp_dst, des_dst = sift.detectAndCompute(gray_img_dst, None)

    # 初始化FLANN基于特征匹配器
    ## 参数？？
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 特征点匹配
    matches = flann.knnMatch(des_src, des_dst, k=2)

    # 保留好的匹配结果
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 获取匹配特征点的坐标
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 对图像进行配准
    aligned_img = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))

    return aligned_img

def main():
    # 读取三张图像
    img1 = cv2.imread('./raw_images/img1.jpg')
    img2 = cv2.imread('./raw_images/img2.jpg')
    img3 = cv2.imread('./raw_images/img3.jpg')
    img4 = cv2.imread('./raw_images/img4.jpg')
    img5 = cv2.imread('./raw_images/img5.jpg')

    # 改成图像数组
    Img = np.array(img1, img2, img3, img4, img5)

    # 计算三张图像的梯度幅值
    ## 改一下avg
    grad_magnitude_img = calculate_average_gradient_magnitude(img1)
    std_img = img1
    # grad_magnitude_img2 = calculate_average_gradient_magnitude(img2)
    # grad_magnitude_img3 = calculate_average_gradient_magnitude(img3)
    # grad_magnitude_img4 = calculate_average_gradient_magnitude(img4)
    # grad_magnitude_img5 = calculate_average_gradient_magnitude(img5)
    for i in range(Img.len()-1):
        grad = calculate_average_gradient_magnitude(Img[i+1])
        if grad > grad_magnitude_img:
            grad_magnitude_img = grad
            std_img = Img[i+1]
            print("最清晰的图片是img" + str(i+1))



    # 找出梯度幅值最大的图像作为基准图像
    ## 用max函数
    # std_img = None
    # if grad_magnitude_img1 > grad_magnitude_img2 and grad_magnitude_img1 > grad_magnitude_img3:
    #     std_img = img1
    #     print("最清晰的图片是img1")
    # elif grad_magnitude_img2 > grad_magnitude_img1 and grad_magnitude_img2 > grad_magnitude_img3:
    #     std_img = img2
    #     print("最清晰的图片是img2")
    # else:
    #     std_img = img3
    #     print("最清晰的图片是img3")

    # # 显示梯度幅值最大的图像
    # cv2.imshow('Maximum Gradient Image', std_img)
    # cv2.waitKey(0)

    # # 关闭所有打开的窗口
    # cv2.destroyAllWindows()

    # Perform image registration using feature matching between img2 and std_img
    # Assuming you have already implemented this function
    registered_img2 = feature_matching(img2, std_img)

    # Perform image registration using feature matching between img3 and std_img
    # Assuming you have already implemented this function
    registered_img3 = feature_matching(img3, std_img)

    registered_img4 = feature_matching(img4, std_img)
    registered_img5 = feature_matching(img5, std_img)

    # Get the current script's directory
    current_dir = os.path.dirname(__file__)

    # Construct the path to the output folder
    registered_images_folder = os.path.join(current_dir, 'reg_images')

    # Assuming img1Reg contains the image data you want to save
    # Save the image in the output folder
    output_path2 = os.path.join(registered_images_folder, 'registered_img2.jpg')
    cv2.imwrite(output_path2, registered_img2)

    output_path3 = os.path.join(registered_images_folder, 'registered_img3.jpg')
    cv2.imwrite(output_path3, registered_img3)
    
    output_path4 = os.path.join(registered_images_folder, 'registered_img4.jpg')
    cv2.imwrite(output_path4, registered_img4)
    output_path5 = os.path.join(registered_images_folder, 'registered_img5.jpg')
    cv2.imwrite(output_path5, registered_img5)

    print(f"图片已存入: {output_path2}")

main()