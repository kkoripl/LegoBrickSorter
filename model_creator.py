import os
import random

import cv2
import numpy as np
from imutils import paths
from keras.applications.mobilenetv2 import MobileNetV2
from keras.callbacks import CSVLogger
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.metrics import top_k_categorical_accuracy
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, label_binarize

from app_params import AppParams
from draw_utils.draw_data import plot_roc_curves_for_classes


class ModelCreator(object):
    categories_cnt = None  # liczba kategorii
    base_model = None
    model = None
    learn_history = None
    labels = None

    train = []  # zbiór testowy
    val = []  # zbiór walidacyjny
    train_val = []  # zbiór trenujący oraz walidacyjny
    test = []  # zbiór testujący

    train_lab = []  # kategorie zbioru trenującego
    val_lab = []  # kategorie zbioru walidacyjnego
    train_val_lab = []  # kategorie zbioru trenującego oraz walidacyjnego
    test_lab = []  # kategorie zbioru testującego

    def read_prepared_data(self, binarize_labels=True, save_labels_for_svm=False):
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

        self.labels = list(set(labels))

        if save_labels_for_svm:
            np.save(AppParams.svm_labels_path, self.labels)

        self.categories_cnt = len(self.labels)  # liczba kategorii
        images = np.array(images)
        if binarize_labels:
            encoded_labels = np.array(self.binarize_labels(labels))  # binaryzacja kategorii
        else:
            dict_labels = dict(zip(self.labels, range(0, len(self.labels))))
            encoded_labels = np.array([dict_labels[label] for label in labels]) # kategorie numerycznie

        # podział całego zbioru obrazów na zbiór testujący i walidacyjno-trenujący
        (self.train_val, self.test, self.train_val_lab, self.test_lab) = train_test_split(images,
                                                                                          encoded_labels,
                                                                                          test_size=AppParams.test_part,
                                                                                          random_state=AppParams.random_state)

        if save_labels_for_svm:
            np.save(AppParams.svm_test_labels_path, self.test_lab)

        # podział zbioru walidacyno-trenującego na zbiór walidacyjny i trenujący
        (self.train, self.val, self.train_lab, self.val_lab) = train_test_split(self.train_val, self.train_val_lab,
                                                                                test_size=AppParams.test_part,
                                                                                random_state=AppParams.random_state)

        if save_labels_for_svm:
            np.save(AppParams.svm_train_labels_path, self.train_lab)

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
        first_removed_layer = base_model_layers_cnt-layers_to_remove_cnt
        first_saved_layer = first_removed_layer-1
        del self.base_model.layers[first_removed_layer:base_model_layers_cnt]  # usuwanie warstw
        for layer in self.base_model.layers:
            layer.trainable = True
        output_tensor = self._add_classification_layers(self.base_model.layers[first_saved_layer],True)
        model = Model(input=self.base_model.input, outputs=output_tensor)
        model = self.compile_model(model, optimizer=optimizer, loss=loss)
        #model.summary() podgląd jak wyglada model
        self.model, self.learn_history = self.fit_model(model, final_training_mode)

    def fit_model(self, model, training_mode, out_file_name=None):
        if out_file_name is None:
            csv_logger = CSVLogger('reports/current.log')
        else:
            csv_logger = CSVLogger(out_file_name)

        if training_mode:
            learn_history = model.fit(
                x=self.train_val,
                y=self.train_val_lab,
                verbose=1,
                callbacks=[csv_logger],
                epochs=AppParams.epochs,
                validation_data=(self.test, self.test_lab)
            )
        else:
            learn_history = model.fit(
                x=self.train,
                y=self.train_lab,
                verbose=1,
                callbacks=[csv_logger],
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
                      metrics=['categorical_accuracy', top_k_categorical_accuracy_metric])
        return model


    def _add_classification_layers(self, base_model, training_whole_model=False):
        standard_size_layer = base_model.output
        # Przy eksperymentach z uczeniem całej sieci potrzebne jest nagłe zmienienie wymiaru, aby klasyfikator dawał wynik w postaci liczby klas
        # Zbijamy do 1D, ponieważ dense działa tylko na ostatnim wymiarze (zmienia go na liczbę swoich neuronow), a my mamy 20 klas (1d)
        if training_whole_model:
            standard_size_layer = base_model.output
            # Wymiary warstwy zwracane są np. tak: (none, 1,2,3) mimo iz faktycznie zwracane jest 3D, w length daje to 4 wymiary
            if len(base_model.output.shape) == 4:
                # Pooling w 2D: Robimy z 3d => 1d
                standard_size_layer = GlobalAveragePooling2D()(base_model.output)
            elif len(base_model.output.shape) == 3:
                # Pooling w 2D: Robimy z 2d => 1d
                standard_size_layer = GlobalAveragePooling1D()(base_model.output)

        first_classif_layer = Dense(AppParams.first_classif_layer_size, activation='relu')(standard_size_layer)
        dropout_layer = Dropout(AppParams.dropout_rate)(first_classif_layer)
        output_tensor = Dense(self.categories_cnt, activation='softmax')(dropout_layer)
        return output_tensor

    # binaryzacja kategorii
    @staticmethod
    def binarize_labels(labels):
        lb = LabelBinarizer()
        encoded_labels = lb.fit_transform(labels)
        return encoded_labels


def top_k_categorical_accuracy_metric(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, AppParams.top_k)

def compute_roc_curves(test_labels, labels_probs, categories_cnt, labels, subtitle, plot_idx):
    test_labels = label_binarize(test_labels, classes=[0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(categories_cnt):
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], labels_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plot_roc_curves_for_classes(fpr, tpr, roc_auc, labels, 4, subtitle, plot_idx)
