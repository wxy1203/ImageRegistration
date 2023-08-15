import os
import cv2
import numpy as np
import img_reg
import video_make

def main():

    img_reg.image_reg('./reg_images', './raw_images')
    video_make.MakeVideo('./output_video.mp4', './reg_images')

main()