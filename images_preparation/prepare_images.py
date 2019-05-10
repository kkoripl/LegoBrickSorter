import os

from PIL import Image

from AppParameters import AppParameters
from data import Datasets


def prepare_images():
    images_dir = os.path.abspath(AppParameters.cropped_img_dir)
    prep_img_dir = os.path.abspath(AppParameters.prep_img_dir)
    brick_dirs = get_brick_dirs()

    max_dimension = find_biggest_img_dimension()
    make_prep_img_dirs_as(prep_img_dir, images_dir)

    for brick_dir in brick_dirs:
        resized_brick_dir = os.path.join(prep_img_dir, brick_dir)
        brick_dir = os.path.join(images_dir, brick_dir)

        pictures = os.listdir(brick_dir)
        resize_and_save_images(max_dimension, pictures, brick_dir, resized_brick_dir)
        print('{} resized'.format(brick_dir))


def resize_and_save_images(new_dimension, images_paths, actual_dir, target_dir):
    dim = (new_dimension[0], new_dimension[1])
    for img_path in images_paths:
        img = Image.open(os.path.join(actual_dir, img_path))
        layer = Image.new('RGB', new_dimension, (255, 255, 255))
        layer.paste(img, tuple(map(lambda x: round((x[0] - x[1]) / 2), zip(dim, img.size))))
        layer = layer.resize(AppParameters.img_size, resample=Image.ANTIALIAS)
        layer.save(os.path.join(target_dir, img_path))

    #todo: brzydko wygladaja obrazki z biala obwodka, a dodatkowo maja slaba jakosc
    #todo: ewentualne samo skalowanie do jakiejs tam wielkosci w kodzie ponizej - ono wyglada lepiej
        # cv2.imwrite(os.path.join(target_dir, img_path),
        #             cv2.resize(cv2.imread(os.path.join(actual_dir, img_path), cv2.IMREAD_COLOR),
        #            AppParameters.img_size,
        #            interpolation=cv2.INTER_CUBIC))


def get_brick_dirs():
    return [subdir for subdir in os.listdir(os.path.abspath(AppParameters.cropped_img_dir))]


def make_prep_img_dirs_as(main_resize_dir, main_dir):
    os.makedirs(main_resize_dir, exist_ok=True)

    for resize_subdir in [subdir for subdir in os.listdir(main_dir)]:
        os.makedirs(os.path.join(main_resize_dir, resize_subdir), exist_ok=True)


def find_biggest_img_dimension():
    max_dimension = [0, 0]
    for root, dirs, imgs in os.walk(AppParameters.cropped_img_dir, topdown=False):
        for img in imgs:
            img_dim = Image.open(os.path.join(root, img)).size
            if img_dim[0] > max_dimension[0]:
                max_dimension[0] = img_dim[0]
            if img_dim[1] > max_dimension[1]:
                max_dimension[1] = img_dim[1]

    return max_dimension

def test():
    datasets = Datasets(0.75)