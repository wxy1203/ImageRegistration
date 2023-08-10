import cv2
import numpy as np


def match_keypoints(src_img, dst_img):
    # 使用SIFT算法进行特征点检测和匹配
    sift = cv2.SIFT_create()
    ini_kp_src, des_src = sift.detectAndCompute(src_img, None)
    ini_kp_dst, des_dst = sift.detectAndCompute(dst_img, None)

    # 使用FLANN匹配器进行特征点匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    rawSrcQrySet = flann.knnMatch(des_src, des_dst, k=2)

    # 保留匹配度较高的特征点
    good_matches = []
    for m, n in rawSrcQrySet:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 提取匹配点的坐标s
    #src_pts_gd = np.float32([raw_kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #dst_pts_gd = np.float32([raw_kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    src_pts_gd = np.float32([ini_kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts_gd = np.float32([ini_kp_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # src_pts_rw = np.float32([ini_kp_src[m.queryIdx].pt for m in raw_matches]).reshape(-1, 1, 2)
    # dst_pts_rw = np.float32([ini_kp_dst[m.trainIdx].pt for m in raw_matches]).reshape(-1, 1, 2)

    # 使用RANSAC算法找到最优的单应性矩阵
    M, mask_us = cv2.findHomography(src_pts_gd, dst_pts_gd, cv2.RANSAC, 5.0)

    good_kp_src = []
    good_kp_dst = []
    for gm in good_matches:
        good_kp_src.append(ini_kp_src[gm.queryIdx])
        good_kp_dst.append(ini_kp_dst[gm.trainIdx])

    raw_kp_src = []
    raw_kp_dst = []
    for m, n in rawSrcQrySet:
        raw_kp_src.append(ini_kp_src[m.queryIdx])
        raw_kp_dst.append(ini_kp_dst[m.trainIdx])  # m的distance更小，忽略n

    return M, mask_us, good_kp_src, good_kp_dst, raw_kp_src, raw_kp_dst


def draw_points(img, pts, r, color, p_style):
    for p in pts:
        x = int(p[0]+0.5)
        y = int(p[1]+0.5)
        if 0 == p_style:
            cv2.circle(img, (x, y), r, color, -1)
        elif 1 == p_style:
            cv2.line(img, (x-r, y-r), (x+r, y+r), color)
            cv2.line(img, (x-r, y+r), (x+r, y-r), color)
    return img


def draw_lines(img, pts, map_pts, color):
    n_Ln = min(len(pts), len(map_pts))
    for i in range(n_Ln):
        if i % 50 == 1000:
            continue
        else:
            x = int(pts[i][0] + 0.5)
            y = int(pts[i][1] + 0.5)
            xm = int(map_pts[i][0] + 0.5)
            ym = int(map_pts[i][1] + 0.5)
            # if abs(x - xm) < 100 and abs(y - ym) < 100:
            cv2.line(img, (x, y), (xm, ym), color)
    return img

def draw_pt_raw_gd_us(img, b2gray, pt_rw, pt_gd, pt_us, p_style):
    if 1 == b2gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if len(img.shape) == 2:
        img_clr = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for i in range(img_clr.shape[0]):
            for j in range(img_clr.shape[1]):
                for k in range(img_clr.shape[2]):
                    img_clr[i][j][k] = img[i][j]
        img = img_clr

    img = draw_points(img, pt_rw, 2, (255, 0, 0), p_style)  # BGR
    img = draw_points(img, pt_gd, 2, (0, 255, 0), p_style)
    img = draw_points(img, pt_us, 2, (0, 0, 255), p_style)
    return img


def draw_df_raw_gd_us(img, b2gray,
                      pt_rw, pt_gd, pt_us,
                      map_pt_rw, map_pt_gd, map_pt_us):
    if 1 == b2gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if len(img.shape) == 2:
        img_clr = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for i in range(img_clr.shape[0]):
            for j in range(img_clr.shape[1]):
                for k in range(img_clr.shape[2]):
                    img_clr[i][j][k] = img[i][j]
        img = img_clr

    # img = draw_lines(img, pt_rw, map_pt_rw, (255, 0, 0))  # BGR
    img = draw_lines(img, pt_gd, map_pt_gd, (0, 255, 0))
    img = draw_lines(img, pt_us, map_pt_us, (0, 0, 255))
    return img


if __name__ == "__main__":
    dst_img = cv2.imread("./raw_images/img1.jpg")
    src_img = cv2.imread("./raw_images/img2.jpg")

    M, mask_us, kp_src_gd, kp_dst_gd, kp_src_rw, kp_dst_rw = match_keypoints(src_img, dst_img)

    MI = np.linalg.inv(M)

    pt_src_rw = [kp_src_rw[i].pt for i in range(len(kp_src_rw))]
    pt_dst_rw = [kp_dst_rw[i].pt for i in range(len(kp_dst_rw))]

    pt_src_gd = [kp_src_gd[i].pt for i in range(len(kp_src_gd))]
    pt_dst_gd = [kp_dst_gd[i].pt for i in range(len(kp_dst_gd))]

    pt_src_us = [kp_src_gd[i].pt for i, m in enumerate(mask_us) if m == 1]
    pt_dst_us = [kp_dst_gd[i].pt for i, m in enumerate(mask_us) if m == 1]

    map_pt_src_rw = [0 for i in range(len(kp_src_rw))]
    map_pt_src_gd = [0 for i in range(len(kp_src_gd))]
    map_pt_src_us = [0 for i in range(len(pt_src_us))]

    map_pt_dst_rw = [0 for i in range(len(kp_dst_rw))]
    map_pt_dst_gd = [0 for i in range(len(kp_dst_gd))]
    map_pt_dst_us = [0 for i in range(len(pt_dst_us))]

    # for i in range(len(pt_src_rw)):
    #     map_pt_src_rw[i] = np.dot(M, (pt_src_rw[i][0], pt_src_rw[i][1], 0))
    # for i in range(len(pt_src_gd)):
    #     map_pt_src_gd[i] = np.dot(M, (pt_src_gd[i][0], pt_src_gd[i][1], 0))
    # for i in range(len(pt_src_us)):
    #     map_pt_src_us[i] = np.dot(M, (pt_src_us[i][0], pt_src_us[i][1], 0))

    for i in range(len(pt_dst_rw)):
        map_pt_dst_rw[i] = np.dot(MI, (pt_dst_rw[i][0], pt_dst_rw[i][1], 0))
    for i in range(len(pt_dst_gd)):
        map_pt_dst_gd[i] = np.dot(MI, (pt_dst_gd[i][0], pt_dst_gd[i][1], 0))
    for i in range(len(pt_dst_us)):
        map_pt_dst_us[i] = np.dot(MI, (pt_dst_us[i][0], pt_dst_us[i][1], 0))

    # for i in range(len(pt_src_rw)):
    #     map_pt_src_rw[i] = np.dot(MI, (pt_src_rw[i][0], pt_src_rw[i][1], 0))
    # for i in range(len(pt_src_gd)):
    #     map_pt_src_gd[i] = np.dot(MI, (pt_src_gd[i][0], pt_src_gd[i][1], 0))
    # for i in range(len(pt_src_us)):
    #     map_pt_src_us[i] = np.dot(MI, (pt_src_us[i][0], pt_src_us[i][1], 0))

    img_src_dst_cmp = src_img.copy()

    src_img = draw_pt_raw_gd_us(src_img, 1, pt_src_rw, pt_src_gd, pt_src_us, 0)
    dst_img = draw_pt_raw_gd_us(dst_img, 1, pt_dst_rw, pt_dst_gd, pt_dst_us, 0)
    # cv2.imshow("Dst Image Registration", dst_img)
    # cv2.waitKey(0)
    img_src_dst_cmp = draw_pt_raw_gd_us(img_src_dst_cmp, 1, map_pt_src_rw, map_pt_src_gd, map_pt_src_us, 1)
    img_src_dst_cmp = draw_pt_raw_gd_us(img_src_dst_cmp, 0,  pt_src_rw, pt_src_gd, pt_src_us, 0)

    # n_fine = 5
    # pt_fine_rw = pt_dst_rw * n_fine
    # pt_fine_gd = pt_dst_gd * n_fine
    # pt_fine_us = pt_dst_us * n_fine
    # map_pt_fine_rw = map_pt_src_rw * n_fine
    # map_pt_fine_gd = map_pt_src_gd * n_fine
    # map_pt_fine_us = map_pt_src_us * n_fine
    ## raw 不成对； 不能用相同序号
    # img_cmp_fine = cv2.resize(img_src_dst_cmp, (img_src_dst_cmp.shape[0] * n_fine, img_src_dst_cmp.shape[1] * n_fine))
    img_src_dst_cmp = draw_df_raw_gd_us(img_src_dst_cmp, 0,
                                        pt_src_rw, pt_src_gd, pt_src_us,
                                        map_pt_dst_rw, map_pt_dst_gd, map_pt_dst_us)

    # cv2.imshow("Src Image Registration", src_img)
    # cv2.waitKey(0)
    #
    # cv2.imshow("Dst Image Registration", dst_img)
    # cv2.waitKey(0)

    cv2.imshow("Cmp Image Registration", img_src_dst_cmp)
    cv2.waitKey(0)

    # 拼接两张图片以展示结果
    result_image = np.hstack((src_img, dst_img))

    # 显示配准结果
    # cv2.imshow("Image Registration", result_image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
