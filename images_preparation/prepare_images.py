import os

from PIL import Image

from AppParameters import AppParameters
from data import Datasets


def prepare_images():
    images_dir = os.path.abspath(AppParameters.cropped_img_dir)
    prep_img_dir = os.path.abspath(AppParameters.prep_img_dir)
    brick_dirs = get_brick_dirs()

    make_prep_img_dirs_as(prep_img_dir, images_dir)

    for brick_dir in brick_dirs:
        resized_brick_dir = os.path.join(prep_img_dir, brick_dir)
        brick_dir = os.path.join(images_dir, brick_dir)

        pictures = os.listdir(brick_dir)
        resize_and_save_images(pictures, brick_dir, resized_brick_dir)
        print('{} resized'.format(brick_dir))


def resize_and_save_images(images_paths, actual_dir, target_dir):
    for img_path in images_paths:
        img = Image.open(os.path.join(actual_dir, img_path))
        if img.size[0] > img.size[1]:
            new_dim = (img.size[0], img.size[0])
        else:
            new_dim = (img.size[1], img.size[1])
        layer = Image.new('RGB', new_dim, (255, 255, 255))
        layer.paste(img, tuple(map(lambda x: round((x[0] - x[1]) / 2), zip(new_dim, img.size))))
        layer = layer.resize(AppParameters.img_size, resample=Image.ANTIALIAS)
        layer.save(os.path.join(target_dir, img_path))


def get_brick_dirs():
    return [subdir for subdir in os.listdir(os.path.abspath(AppParameters.cropped_img_dir))]


def make_prep_img_dirs_as(main_resize_dir, main_dir):
    os.makedirs(main_resize_dir, exist_ok=True)

    for resize_subdir in [subdir for subdir in os.listdir(main_dir)]:
        os.makedirs(os.path.join(main_resize_dir, resize_subdir), exist_ok=True)


def test():
    datasets = Datasets(0.75)

prepare_images()