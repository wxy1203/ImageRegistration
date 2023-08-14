import cv2
import os

# 设置输入图像文件夹和输出视频文件名
input_folder = './reg_images'
output_video = 'output_video.mp4'

# 获取文件夹中的所有图像文件
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')])

# 获取第一张图像的尺寸，作为视频的尺寸
first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
height, width, layers = first_image.shape

# 设置视频编解码器和帧速率
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以尝试其他编解码器，如XVID
fps = 30  # 帧速率

# 创建视频写入器对象
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 遍历所有图像文件，将每张图像写入视频
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    video_writer.write(image)

# 释放视频写入器
video_writer.release()

print("视频已创建：", output_video)
