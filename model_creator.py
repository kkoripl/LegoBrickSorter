import os
import random

import cv2
import numpy as np
from imutils import paths
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Input
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from app_params import AppParams


class ModelCreator(object):
    categories_cnt = None
    base_model = None
    model = None
    learn_history = None

    train = []
    val = []
    train_val = []
    test = []

    train_lab = []
    val_lab = []
    train_val_lab = []
    test_lab = []

    def read_prepared_data(self):
        image_paths = sorted(list(paths.list_images('data_prep')))
        random.seed(AppParams.random_state)
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
        images = np.array(images)
        encoded_labels = np.array(self.binarize_labels(labels))

        (self.train_val, self.test, self.train_val_lab, self.test_lab) = train_test_split(images,
                                                                                          encoded_labels,
                                                                                          test_size=AppParams.test_part,
                                                                                          random_state=AppParams.random_state)

        (self.train, self.val, self.train_lab, self.val_lab) = train_test_split(self.train_val, self.train_val_lab,
                                                                                test_size=AppParams.test_part,
                                                                                random_state=AppParams.random_state)

    def load_base_pretrained_model(self):
        model_exists = os.path.isfile(AppParams.base_model_path)
        if model_exists:
            self.base_model = load_model(AppParams.base_model_path)
        else:
            self.base_model = self._download_pretrained_model()

    def _download_pretrained_model(self):
        input_tensor = Input(shape=(AppParams.img_size[0], AppParams.img_size[1], 3))
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor,
            input_shape=(AppParams.img_size[0], AppParams.img_size[1], 3),
            pooling='avg')
        base_model.save(AppParams.base_model_path)
        return base_model

    def train_last_fully_connected_layer(self, optimizer=Adam(), loss=AppParams.loss, final_training_mode=True):
        for layer in self.base_model.layers:
            layer.trainable = False
        output_tensor = self._add_output_layer(self.base_model)
        model = Model(input=self.base_model.input, outputs=output_tensor)
        model = self.compile_model(model, optimizer=optimizer, loss=loss)
        self.model, self.learn_history = self.fit_model(model, final_training_mode)

    def train_from_last_convolutional_layer(self, optimizer=Adam(), loss=AppParams.loss, final_training_mode=True):
        first_conv_trainable_set = False
        for layer in reversed(self.base_model.layers):
            layer.trainable = not first_conv_trainable_set
            if type(layer) is Conv2D:
                first_conv_trainable_set = True
        output_tensor = self._add_output_layer(self.base_model)
        model = Model(input=self.base_model.input, outputs=output_tensor)
        model = self.compile_model(model, optimizer=optimizer, loss=loss)
        self.model, self.learn_history = self.fit_model(model, final_training_mode)

    def fit_model(self, model, training_mode):
        if training_mode:
            learn_history = model.fit(
                x=self.train_val,
                y=self.train_val_lab,
                verbose=1,
                epochs=AppParams.epochs,
                validation_data=(self.test, self.test_lab)
            )
        else:
            learn_history = model.fit(
                x=self.train,
                y=self.train_lab,
                verbose=1,
                epochs=AppParams.val_epochs,
                validation_data=(self.val, self.val_lab)
            )
        return model, learn_history

    def train_last_fully_connected_layer_further(self, initial_epoch=None):
        model = load_model(AppParams.trained_model_path)
        for layer in model.layers:
            layer.trainable = False
        model.layers[-1].trainable = True

        learn_history = model.fit(
            x=self.train_val,
            y=self.train_val_lab,
            verbose=1,
            epochs=AppParams.steps_per_epoch,
            validation_data=(self.test, self.test_lab),
            initial_epoch=initial_epoch
        )
        self.model = model
        return learn_history

    def save_model(self, path):
        self.model.save(path)

    def compile_model(self, model, optimizer, loss):
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['categorical_accuracy'])
        return model

    def _add_output_layer(self, base_model):
        output_tensor = Dense(self.categories_cnt, activation='softmax')(base_model.output)
        output_tensor.trainable = True
        return output_tensor

    @staticmethod
    def binarize_labels(labels):
        lb = LabelBinarizer()
        encoded_labels = lb.fit_transform(labels)
        return encoded_labels

