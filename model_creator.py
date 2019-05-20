import os
import random

import cv2
import numpy as np
from imutils import paths
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Input
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from app_params import AppParams


class ModelCreator(object):
    images = []
    encoded_labels = []
    categories_cnt = None
    base_model = None

    def read_prepared_data(self):
        image_paths = sorted(list(paths.list_images('data_prep')))
        random.seed(37)
        random.shuffle(image_paths)
        examples_cnt = len(image_paths)
        images = []
        labels = []

        for idx, imagePath in enumerate(image_paths):
            image = cv2.imread(imagePath)
            normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            print('image {} of {} read'.format(idx, examples_cnt))
            images.append(normalized_image)
            label = imagePath.split(os.path.sep)[-2]
            labels.append(label)

        self.categories_cnt = len(set(labels))
        self.images = np.array(images)
        self.encoded_labels = np.array(self.binarize_labels(labels))

    def load_base_pretrained_model(self):
        model_exists = os.path.isfile(AppParams.base_model_path)
        if model_exists:
            self.base_model = load_model(AppParams.base_model_path)
        else:
            self.base_model = self.download_pretrained_model()
        for layer in self.base_model.layers:
            layer.trainable = False

    def download_pretrained_model(self):
        input_tensor = Input(shape=(AppParams.img_size[0], AppParams.img_size[1], 3))
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor,
            input_shape=(AppParams.img_size[0], AppParams.img_size[1], 3),
            pooling='avg')
        base_model.save(f'models/base_model_mobilenetv2_pool_avg_{AppParams.img_size[0]}.h5')
        return base_model

    def train_last_fully_connected_layer(self):
        output_tensor = self._add_output_layer(self.base_model)
        model = Model(input=self.base_model.input, outputs=output_tensor)

        model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        print('model compiled')

        (train, test, train_lab, test_lab) = train_test_split(self.images,
                                                              self.encoded_labels, test_size=0.25, random_state=42)

        steps_per_epoch = 1
        epochs = 50
        model.fit(
            x=train,
            y=train_lab,
            steps_per_epoch=steps_per_epoch,
            verbose=1,
            epochs=epochs)

        model.save(f'models/last_layer_trained_mobilenetv2_{steps_per_epoch}_{epochs}.h5')



    def train_last_fully_connected_layer_further(self):
        model = load_model(AppParams.trained_model_path)
        for layer in model.layers:
            layer.trainable = False

        model.layers[-1].trainable = True

        (train, test, train_lab, test_lab) = train_test_split(self.images,
                                                              self.encoded_labels, test_size=0.25, random_state=42)

        steps_per_epoch = 1
        epochs = 20
        model.fit(
            x=train,
            y=train_lab,
            steps_per_epoch=steps_per_epoch,
            verbose=1,
            epochs=epochs)

        model.save(f'models/trained_further_last_layer_mobilenetv2_{epochs}.h5')


    def _add_output_layer(self, base_model):
        output_tensor = Dense(self.categories_cnt, activation='softmax')(base_model.output)
        output_tensor.trainable = True
        return output_tensor

    @staticmethod
    def binarize_labels(labels):
        lb = LabelBinarizer()
        encoded_labels = lb.fit_transform(labels)
        return encoded_labels


model_creator = ModelCreator()
model_creator.load_base_pretrained_model()
model_creator.read_prepared_data()
model_creator.train_last_fully_connected_layer_further()

