import cv2

# Load the image
image_path = "./raw_images/img1.jpg"
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a keypoint detector (e.g., SIFT or SURF)
detector = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = detector.detectAndCompute(gray_image, None)

# Draw the keypoints on the image
output_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with key points
cv2.imshow("Image with Key Points", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()