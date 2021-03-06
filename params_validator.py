from model_creator import ModelCreator
from draw_utils import draw_data as dd
from app_params import AppParams


def validate_optimizers(optimizers):
    for optimizer in optimizers:
        model_creator = ModelCreator()
        model_creator.load_base_pretrained_model()
        model_creator.read_prepared_data()
        model_creator.train_last_fully_connected_layer()


def validate_loss_functions(functions):
    for func in functions:
        model_creator = ModelCreator()
        model_creator.load_base_pretrained_model()
        model_creator.read_prepared_data()
        model_creator.train_last_fully_connected_layer(final_training_mode=False, loss=func)
        dd.draw_learning_history(model_creator.learn_history, f'Funkcja kary: {func}')


def validate_first_classif_sizes(sizes):
    learn_histories = []
    for size in sizes:
        model_creator = ModelCreator()
        model_creator.load_base_pretrained_model()
        model_creator.read_prepared_data()
        model_creator.train_last_fully_connected_layer(final_training_mode=False, first_classif_layer_size=size,
                                                       logs_file=f'valid_size_{size}.csv')
        learn_histories.append(model_creator.learn_history)
    dd.draw_used_params_values_comparision(histories=learn_histories, values=sizes, out_filename='sizes_validation')


def validate_dropout_rates(dropout_rates):
    learn_histories = []
    for dropout_rate in dropout_rates:
        model_creator = ModelCreator()
        model_creator.load_base_pretrained_model()
        model_creator.read_prepared_data()
        model_creator.train_last_fully_connected_layer(final_training_mode=False, dropout_rate=dropout_rate,
                                                       logs_file=f'valid_dropout_{dropout_rate}.csv')
        learn_histories.append(model_creator.learn_history)
    dd.draw_used_params_values_comparision(histories=learn_histories, values=dropout_rates,
                                           out_filename='dropout_validation')


def validate_removal_coef(removal_coefs):
    for removal_coef in removal_coefs:
        model_creator = ModelCreator()
        model_creator.load_base_pretrained_model()
        model_creator.read_prepared_data()
        model_creator.train_whole_convolutional_network(removal_coef, final_training_mode=False)
        dd.draw_learning_history(model_creator.learn_history, f'Wspolczynnik usunietych warstw: {removal_coef}')
        model_creator.save_model(
            f'models/{model_creator.base_model.name}_{AppParams.layers_trainable_mode.name}_{AppParams.epochs}_{removal_coef}.h5')
        dd.draw_learning_history(model_creator.learn_history)
        del model_creator


def validate_all(sizes, droput_rates):
    learn_histories_dict = {}
    for row_idx, size in enumerate(sizes):
        for col_idx, rate in enumerate(droput_rates):
            print(f'class_first_size_{size}_dr_rate{rate}')
            model_creator = ModelCreator()
            model_creator.load_base_pretrained_model()
            model_creator.read_prepared_data()
            model_creator.train_last_fully_connected_layer(final_training_mode=True,
                                                           first_classif_layer_size=size,
                                                           dropout_rate=rate,
                                                           logs_file=f'class_first_size_{size}_dr_rate{rate}.csv')
            learn_histories_dict[(size, rate)] = model_creator.learn_history
            dd.draw_learning_history(model_creator.learn_history, title='Uczenie części klasyfikującej',
                                     out_filename=f'class_first_size_{size}_dr_rate{rate}')
            dd.draw_learning_history_top_k(model_creator.learn_history, title='Uczenie części klasyfikującej',
                                           out_filename=f'class_first_size_{size}_dr_rate{rate}')
    dd.draw_two_used_params_values_comparision(histories_dict=learn_histories_dict,
                                               out_filename='sizes_and_rates_first_classif')


def valuate_first_classif_sizes_from_last_conv(sizes):
    learn_histories = []
    for size in sizes:
        model_creator = ModelCreator()
        model_creator.load_base_pretrained_model()
        model_creator.read_prepared_data()
        model_creator.train_from_last_convolutional_layer(final_training_mode=True, first_classif_layer_size=size,
                                                       logs_file=f'valid_size_{size}_last_conv.csv')
        learn_histories.append(model_creator.learn_history)
        dd.draw_learning_history(model_creator.learn_history, title='Uczenie od warstwy splotowej',
                                 out_filename=f'last_conv_size_{size}')
        dd.draw_learning_history_top_k(model_creator.learn_history, title='Uczenie części klasyfikującej',
                                       out_filename=f'last_conv_size_{size}')
    dd.draw_used_params_values_comparision(histories=learn_histories, values=sizes, out_filename='sizes_validation_last_conv')

validate_all(AppParams.first_classif_layer_sizes, AppParams.dropout_rates)
valuate_first_classif_sizes_from_last_conv(AppParams.first_classif_layer_sizes)