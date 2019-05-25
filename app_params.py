from layers_trainable_modes import LayersTrainableMode


class AppParams(object):
    img_dir = "../data"
    base_img_dir = "../data/Base_Images"
    resized_base_img_dir = "../data/Base_Images_Resized"
    cropped_img_dir = "../data/Cropped_Images"
    prep_img_dir = "../data_prep"

    data_dir = prep_img_dir

    img_size = (128, 128)
    img_channels = 3

    random_state = 37

    layers_trainable_mode = LayersTrainableMode.ALL  # rodzaj zadania
    test_part = 0.25
    epochs = 10
    loss = 'categorical_crossentropy'
    first_classif_layer_size = 10
    dropout_rate = 0.1

    val_epochs = 10

    base_model_path = 'models/base_model_mobilenetv2_avg_pool_128.h5'
    trained_model_path = 'models/mobilenetv2_last_lay_979_acc.h5'

    linear_kernel = 'linear'
    square_kernel = 'poly2'
    exp_kernel = 'rbf'
    svm_kernel_types = [linear_kernel, square_kernel, exp_kernel]