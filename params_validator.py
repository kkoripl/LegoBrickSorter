from model_creator import ModelCreator
from draw_utils import draw_data as dd

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

def validate_removal_coef(removal_coefs):
    for removal_coef in removal_coefs:
        model_creator = ModelCreator()
        model_creator.load_base_pretrained_model()
        model_creator.read_prepared_data()
        model_creator.train_whole_convolutional_network(removal_coef, final_training_mode=False)
        dd.draw_learning_history(model_creator.learn_history, f'Wspolczynnik usunietych warstw: {removal_coef}')