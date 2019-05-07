import os
import cv2

from AppParameters import AppParameters


def prepare_images():
    images_dir = AppParameters.cropped_img_dir
    prep_img_dir = AppParameters.prep_img_dir
    brick_dirs = [subdir for subdir in os.listdir(images_dir)]

    make_prep_img_dirs_as(prep_img_dir, images_dir)

    for brick_dir in brick_dirs:
        resized_brick_dir = os.path.join(prep_img_dir, brick_dir)
        brick_dir = os.path.join(images_dir, brick_dir)

        pictures = os.listdir(brick_dir)
        resize_and_save_images(pictures, brick_dir, resized_brick_dir)
        print('{} resized'.format(brick_dir))


def resize_and_save_images(images, actual_dir, target_dir):
    img_height = AppParameters.img_height
    img_width = AppParameters.img_width

    for image in images:
        cv2.imwrite(os.path.join(target_dir, image),
                    cv2.resize(cv2.imread(os.path.join(actual_dir, image), cv2.IMREAD_COLOR),
                   (img_height, img_width),
                   interpolation=cv2.INTER_CUBIC))


def make_prep_img_dirs_as(main_resize_dir, main_dir):
    os.makedirs(main_resize_dir, exist_ok=True)

    for resize_subdir in [subdir for subdir in os.listdir(main_dir)]:
        os.makedirs(os.path.join(main_resize_dir, resize_subdir), exist_ok=True)

prepare_images()