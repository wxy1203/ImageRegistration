
import img_reg_SIFT
import video_make


def main():

    img_reg_SIFT.image_reg('./reg_images', './raw_images')
    video_make.MakeVideo('./output_video.mp4', './reg_images')

main()