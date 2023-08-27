import cv2


def fuse_image(img_reg, img_std):

    h_reg, w_reg = img_reg.shape[:2]
    h_std, w_std = img_std.shape[:2]

    if h_reg != h_std or w_reg != w_std:
        return None

    img_fus = img_std.copy()
    for j in range(img_std.shape[0]):
        for i in range(img_std.shape[1]):
            img_fus[j][i] = img_std[j][i]/2 + img_reg[j][i]/2

    cv2.imshow('img_reg', img_reg)
    cv2.waitKey(0)

    cv2.imshow('img_std', img_std)
    cv2.waitKey(0)

    cv2.imshow('img_fus', img_fus)
    cv2.waitKey(0)
    return img_fus


def fuse_image_texture(img_reg, img_std):

    h_reg, w_reg = img_reg.shape[:2]
    h_std, w_std = img_std.shape[:2]

    if h_reg != h_std or w_reg != w_std:
        return None

    img_reg_gray = img_reg.copy()
    img_reg_gray = cv2.cvtColor(img_reg_gray, cv2.COLOR_BGR2GRAY)
    x_reg = cv2.Sobel(img_reg_gray, cv2.CV_16S, 1, 0)
    y_reg = cv2.Sobel(img_reg_gray, cv2.CV_16S, 0, 1)
    absX_reg = cv2.convertScaleAbs(x_reg)  # 转回uint8
    absY_reg = cv2.convertScaleAbs(y_reg)
    img_reg_tGry = cv2.addWeighted(absX_reg, 0.5, absY_reg, 0.5, 0)
    cv2.imshow("img_reg_tGry", img_reg_tGry)
    cv2.waitKey(0)

    img_reg_tGreen = img_reg.copy()
    for j in range(img_reg_tGreen.shape[0]):
        for i in range(img_reg_tGreen.shape[1]):
            img_reg_tGreen[j][i][0] = 0
            gray_val = img_reg_tGry[j][i]
            if gray_val > 110:
                img_reg_tGreen[j][i][1] = gray_val
            else:
                img_reg_tGreen[j][i][1] = 0
            img_reg_tGreen[j][i][2] = 0
    cv2.imshow("img_reg_tGreen", img_reg_tGreen)
    cv2.waitKey(0)

    img_std_gray = img_std.copy()
    img_std_gray = cv2.cvtColor(img_std_gray, cv2.COLOR_BGR2GRAY)
    x_std = cv2.Sobel(img_std_gray, cv2.CV_16S, 1, 0)
    y_std = cv2.Sobel(img_std_gray, cv2.CV_16S, 0, 1)
    absX_std = cv2.convertScaleAbs(x_std)  # 转回uint8
    absY_std = cv2.convertScaleAbs(y_std)
    img_std_tGry = cv2.addWeighted(absX_std, 0.5, absY_std, 0.5, 0)
    cv2.imshow("img_std_tGry", img_std_tGry)
    cv2.waitKey(0)

    img_std_tRed = img_std.copy()
    for j in range(img_std_tRed.shape[0]):
        for i in range(img_std_tRed.shape[1]):
            img_std_tRed[j][i][0] = 0
            img_std_tRed[j][i][1] = 0
            gray_val = img_std_tGry[j][i]
            if gray_val > 110:
                img_std_tRed[j][i][2] = gray_val
            else:
                img_std_tRed[j][i][2] = 0
    cv2.imshow("img_std_tRed", img_std_tRed)
    cv2.waitKey(0)

    img_tGreenRed = img_std_tRed + img_reg_tGreen
    cv2.imshow("img_tGreenRed", img_tGreenRed)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


img_reg = cv2.imread('/Users/wangxinyi/Desktop/ImageRegistration/reg_images/registered_img2.jpg')
img_std = cv2.imread('/Users/wangxinyi/Desktop/ImageRegistration/raw_images/img1.jpg')

# fuse_image(img_reg, img_std)
fuse_image_texture(img_reg, img_std)
