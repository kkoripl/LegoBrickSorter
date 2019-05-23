from model_creator import ModelCreator
from app_params import AppParams
from layers_trainable_modes import LayersTrainableMode
from draw_utils import draw_data as dd
import params_validator as pv

model_creator = ModelCreator()
model_creator.load_base_pretrained_model()
model_creator.read_prepared_data()

if AppParams.layers_trainable_mode is LayersTrainableMode.ONLY_CLASSIF:
    model_creator.train_last_fully_connected_layer()
    dd.draw_learning_history(model_creator.learn_history, title='Uczona ostatnia warstwa')
elif AppParams.layers_trainable_mode is LayersTrainableMode.FROM_LAST_CONV:
    model_creator.train_from_last_convolutional_layer()
    dd.draw_learning_history(model_creator.learn_history, title='Uczenie od ostatniej warstwy splotowej')
model_creator.save_model(
    f'models/{model_creator.base_model.name}_{AppParams.layers_trainable_mode.name}_{AppParams.epochs}.h5')
dd.draw_learning_history(model_creator.learn_history)

# do wyboru najlepszej funkcji kosztu
# pv.validate_loss_functions(['mean_squared_error', 'categorical_hinge', 'categorical_crossentropy'])
