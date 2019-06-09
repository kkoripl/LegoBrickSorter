import datetime

from keras.engine.saving import load_model

import params_validator as pv
from app_params import AppParams
from draw_utils import draw_data as dd
from layers_trainable_modes import LayersTrainableMode
from model_creator import ModelCreator
from svm_model import train_svm

model_creator = ModelCreator()
model_creator.load_base_pretrained_model()  # załadowanie wytrenowanego już modelu
model_creator.read_prepared_data()  # odczytanie odpowiednio podzielonych danych

# ZADANIE 1
if AppParams.layers_trainable_mode is LayersTrainableMode.ONLY_CLASSIF:
    model_creator.train_last_fully_connected_layer(logs_file='only_classif.csv')
    dd.draw_learning_history(model_creator.learn_history, title='Uczenie części klasyfikującej', out_filename='only_classif')

# ZADANIE 2
elif AppParams.layers_trainable_mode is LayersTrainableMode.FROM_LAST_CONV:
    model_creator.train_from_last_convolutional_layer(logs_file='from_last_conv.csv')
    dd.draw_learning_history(model_creator.learn_history, title='Uczenie od ostatniej warstwy splotowej', out_filename='from_last_conv')

# ZADANIE 3 - na razie na zasadzie walidacji, aby znaleźć optymalny współczynnik usuwania warstw
elif AppParams.layers_trainable_mode is LayersTrainableMode.ALL:
    removal_coefs = [0.99]  # współczynniki usuwania warstw
    pv.validate_removal_coef(removal_coefs)

# ZADANIE 4 - SVM
elif AppParams.layers_trainable_mode is LayersTrainableMode.SVM_REP:
    print(str(datetime.datetime.now()) + ' - start loading model')
    svm = train_svm(load_model(AppParams.trained_model_path), model_creator)

# zapisanie modelu
# model_creator.save_model(
#    f'models/{model_creator.base_model.name}_{AppParams.layers_trainable_mode.name}_{AppParams.epochs}.h5')
# dd.draw_learning_history(model_creator.learn_history)

# do wyboru najlepszej funkcji kosztu
# pv.validate_loss_functions(['mean_squared_error', 'categorical_hinge', 'categorical_crossentropy'])

# pv.validate_first_classif_sizes(AppParams.first_classif_layer_sizes)
# pv.validate_dropout_rates(AppParams.dropout_rates)
