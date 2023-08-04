import cv2
import numpy as np

def match_keypoints(src_img, dst_img):
    # 使用SIFT算法进行特征点检测和匹配
    sift = cv2.SIFT_create()
    raw_kp_src, des_src = sift.detectAndCompute(src_img, None)
    raw_kp_dst, des_dst = sift.detectAndCompute(dst_img, None)

    # 使用FLANN匹配器进行特征点匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des_src, des_dst, k=2)

    # 保留匹配度较高的特征点
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 提取匹配点的坐标
    #src_pts_gd = np.float32([raw_kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #dst_pts_gd = np.float32([raw_kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)



    src_pts_gd_hm = np.float32([raw_kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts_gd_hm = np.float32([raw_kp_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 使用RANSAC算法找到最优的单应性矩阵
    M, mask_us = cv2.findHomography(src_pts_gd_hm, dst_pts_gd_hm, cv2.RANSAC, 5.0)

    good_kp_src = []
    good_kp_dst = []
    for gm in good_matches:
        good_kp_src.append(raw_kp_src[gm.queryIdx].pt)
        good_kp_dst.append(raw_kp_dst[gm.queryIdx].pt)

    return M, mask_us, good_kp_src, good_kp_dst, raw_kp_src, raw_kp_dst

def draw_keypoints(img, keypoints, color):
    for kp in keypoints:
        x = kp.pt[0]
        y = kp.pt[1]
        cv2.circle(img, (x,y), 2, color, -1)

def draw_kp_raw_gd_us(img, raw_kp, good_kp, use_kp):
    gray_img_dst_w_kp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_keypoints(gray_img_dst_w_kp, raw_kp, (0, 0, 255))
    draw_keypoints(gray_img_dst_w_kp, good_kp, (0, 255, 0))
    draw_keypoints(gray_img_dst_w_kp, use_kp , (255, 0 , 0))


if __name__ == "__main__":
    dst_img = cv2.imread("./raw_images/img1.jpg", cv2.IMREAD_GRAYSCALE)
    src_img = cv2.imread("./raw_images/img2.jpg", cv2.IMREAD_GRAYSCALE)

    M, mask_us, src_pts_gd, dst_pts_gd, raw_kp_src, raw_kp_dst = match_keypoints(src_img, dst_img)

    src_use_kp = [src_pts_gd[i] for i, m in enumerate(mask_us) if m == 1]
    draw_kp_raw_gd_us(src_img, raw_kp_src, src_pts_gd, src_use_kp)

    dst_use_kp = [dst_pts_gd[i] for i, m in enumerate(mask_us) if m == 1]
    draw_kp_raw_gd_us(dst_img, raw_kp_dst, dst_pts_gd, dst_use_kp)

    map_src_kp_raw = np.dot(M, raw_kp_src)
    map_src_kp_good = np.dot(M, src_pts_gd)
    map_src_kp_use = np.dot(M, src_use_kp)

    img_src_dst_cmp = dst_img
    draw_kp_raw_gd_us(img_src_dst_cmp, map_src_kp_raw, map_src_kp_good, map_src_kp_use)


    # img2_with_keypoints = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    # matched_kp2 = [kp2[i] for i, m in enumerate(mask) if m == 1]
    # draw_keypoints(img2_with_keypoints, matched_kp2, (0, 0, 255))  # 用蓝色圈出

    # 拼接两张图片以展示结果
    result_image = np.hstack((src_img, dst_img, img_src_dst_cmp))

    # 显示配准结果
    cv2.imshow("Image Registration", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
