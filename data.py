import os

import cv2

from AppParameters import AppParameters


class Datasets:
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    possible_labels = []

    __data_dir = os.path.abspath(AppParameters.data_dir)

    def __init__(self, pct_of_trainset):
        self.__find_possible_labels()
        self.__find_datasets_files(pct_of_trainset)
        self.__load_datasets_imgs()

    def __find_possible_labels(self):
        self.possible_labels = self.__get_brick_dirs()

    def __get_brick_dirs(self):
        return [subdir for subdir in os.listdir(self.__data_dir)]

    def __find_datasets_files(self, pct_of_trainset):
        brick_dirs = self.__get_brick_dirs()
        for brick_dir in brick_dirs:
            images = [os.path.join(self.__data_dir, brick_dir, img_path)
                      for img_path in os.listdir(os.path.join(self.__data_dir, brick_dir))]
            new_train_imgs_cnt = round(len(images)*pct_of_trainset)
            new_test_imgs_cnt = len(images)-new_train_imgs_cnt
            self.train_imgs += images[:new_train_imgs_cnt]
            self.train_labels += [brick_dir] * new_train_imgs_cnt
            self.test_imgs += images[-new_test_imgs_cnt:]
            self.test_labels += [brick_dir] * new_test_imgs_cnt

    def __load_datasets_imgs(self):
        self.train_imgs = [cv2.imread(img_path) for img_path in self.train_imgs]
        self.test_imgs = [cv2.imread(img_path) for img_path in self.test_imgs]