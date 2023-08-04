import cv2
import numpy as np

def match_keypoints(img1, img2):
    # 使用ORB算法进行特征点检测和匹配
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 使用BFMatcher进行特征点匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 根据距离排序并保留前50个匹配
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]

    # 提取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 使用RANSAC算法找到最优的单应性矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M, mask, kp1, kp2

def draw_keypoints(img, keypoints, color):
    for kp in keypoints:
        x, y = np.int32(kp.pt)
        cv2.circle(img, (x, y), 5, color, -1)

if __name__ == "__main__":
    img1 = cv2.imread("./raw_images/img1.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./raw_images/img2.jpg", cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error: Unable to read image(s).")
        exit(1)

    M, mask, kp1, kp2 = match_keypoints(img1, img2)

    # 在img1中绘制所有特征点
    img1_with_keypoints = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    draw_keypoints(img1_with_keypoints, kp1, (255, 0, 0))  # 用红色圈出

    # 在img2中绘制匹配上的特征点
    img2_with_keypoints = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    matched_kp2 = [kp2[i] for i, m in enumerate(mask) if m == 1]
    draw_keypoints(img2_with_keypoints, matched_kp2, (0, 0, 255))  # 用蓝色圈出

    # 拼接两张图片以展示结果
    result_image = np.hstack((img1_with_keypoints, img2_with_keypoints))

    # 显示配准结果
    cv2.imshow("Image Registration", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
