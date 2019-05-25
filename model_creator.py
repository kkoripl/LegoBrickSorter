import os
import random

import cv2
import numpy as np
from imutils import paths
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Input, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from app_params import AppParams


class ModelCreator(object):
    categories_cnt = None  # liczba kategorii
    base_model = None
    model = None
    learn_history = None

    train = []  # zbiór testowy
    val = []  # zbiór walidacyjny
    train_val = []  # zbiór trenujący oraz walidacyjny
    test = []  # zbiór testujący

    train_lab = []  # kategorie zbioru trenującego
    val_lab = []  # kategorie zbioru walidacyjnego
    train_val_lab = []  # kategorie zbioru trenującego oraz walidacyjnego
    test_lab = []  # kategorie zbioru testującego

    def read_prepared_data(self):
        image_paths = sorted(list(paths.list_images('data_prep')))  # pobranie obrazów
        random.seed(AppParams.random_state)  # ziarno losowe
        random.shuffle(image_paths)  # pomieszanie obrazów
        examples_cnt = len(image_paths)  # liczba obrazów
        images = []  # lista obrazów
        labels = []  # lista kategorii

        # odczytywanie kolejnych obrazów
        for idx, imagePath in enumerate(image_paths):
            image = cv2.imread(imagePath)  # odczytanie obrazu
            normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # normalizacja obrazu
            print('image {} of {} read'.format(idx, examples_cnt))
            images.append(normalized_image)  # dodanie obrazu do listy
            label = imagePath.split(os.path.sep)[-2]  # odczytanie kategorii obrazu
            labels.append(label)  # dodanie kategorii obrazu do listy

        self.categories_cnt = len(set(labels))  # liczba kategorii
        images = np.array(images)
        encoded_labels = np.array(self.binarize_labels(labels))  # binaryzacja kategorii

        # podział całego zbioru obrazów na zbiór testujący i walidacyjno-trenujący
        (self.train_val, self.test, self.train_val_lab, self.test_lab) = train_test_split(images,
                                                                                          encoded_labels,
                                                                                          test_size=AppParams.test_part,
                                                                                          random_state=AppParams.random_state)

        # podział zbioru walidacyno-trenującego na zbiór walidacyjny i trenujący
        (self.train, self.val, self.train_lab, self.val_lab) = train_test_split(self.train_val, self.train_val_lab,
                                                                                test_size=AppParams.test_part,
                                                                                random_state=AppParams.random_state)

    # załadowanie wytrenowanego już modelu
    def load_base_pretrained_model(self):
        model_exists = os.path.isfile(AppParams.base_model_path)  # sprawdzenie czy model jest zapisany na dysku
        if model_exists:
            self.base_model = load_model(AppParams.base_model_path)  # odczytanie modelu z dysku
        else:
            self.base_model = self._download_pretrained_model()  # odczytanie ze strony internetowej

    # odczytanie wytrenowanego modelu ze strony internetowej
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

    # trenowanie ostatniej warstwy klasyfikującej - ZADANIE 1
    def train_last_fully_connected_layer(self, optimizer=Adam(), loss=AppParams.loss, final_training_mode=True):
        for layer in self.base_model.layers:
            layer.trainable = False
        output_tensor = self._add_classification_layers(self.base_model)
        model = Model(input=self.base_model.input, outputs=output_tensor)  # złożenie całkowitego modelu
        model = self.compile_model(model, optimizer=optimizer, loss=loss)
        self.model, self.learn_history = self.fit_model(model, final_training_mode)  # nauka

    # trenowanie również ostatniej warstwy splotowej - ZADANIE 2
    def train_from_last_convolutional_layer(self, optimizer=Adam(), loss=AppParams.loss, final_training_mode=True):
        first_conv_trainable_set = False
        for layer in reversed(self.base_model.layers):
            layer.trainable = not first_conv_trainable_set
            if type(layer) is Conv2D:
                first_conv_trainable_set = True
        output_tensor = self._add_classification_layers(self.base_model)
        model = Model(input=self.base_model.input, outputs=output_tensor)
        model = self.compile_model(model, optimizer=optimizer, loss=loss)
        self.model, self.learn_history = self.fit_model(model, final_training_mode)

    # trenowanie wszystkich warstw w sieci splotowej - ZADANIE 3
    def train_whole_convolutional_network(self, removal_coef, optimizer=Adam(), loss=AppParams.loss, final_training_mode=True):
        base_model_layers_cnt = len(self.base_model.layers)  # liczba warstw splotowych
        layers_to_remove_cnt = round(removal_coef*base_model_layers_cnt)  # liczba warstw do usunięcia
        del self.base_model.layers[base_model_layers_cnt-layers_to_remove_cnt:base_model_layers_cnt]  # usuwanie warstw
        for layer in self.base_model.layers:
            layer.trainable = True
        output_tensor = self._add_classification_layers(self.base_model)
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

    # dotrenowanie, jeżeli jest szansa, że poprawi to aktualny stan
    def train_last_fully_connected_layer_further(self, initial_epoch=None):
        model = load_model(AppParams.trained_model_path)
        for layer in model.layers:
            layer.trainable = False
        model.layers[-1].trainable = True

        learn_history = model.fit(
            x=self.train_val,
            y=self.train_val_lab,
            verbose=1,
            epochs=AppParams.epochs,
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

    def _add_classification_layers(self, base_model):
        first_classif_layer = Dense(AppParams.first_classif_layer_size, activation='relu')(base_model.output)
        dropout_layer = Dropout(AppParams.dropout_rate)(first_classif_layer)
        output_tensor = Dense(self.categories_cnt, activation='softmax')(dropout_layer)
        return output_tensor

    # binaryzacja kategorii
    @staticmethod
    def binarize_labels(labels):
        lb = LabelBinarizer()
        encoded_labels = lb.fit_transform(labels)
        return encoded_labels
