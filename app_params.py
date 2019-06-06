from layers_trainable_modes import LayersTrainableMode


class AppParams(object):
    img_dir = "../data"
    base_img_dir = "../data/Base_Images"
    resized_base_img_dir = "../data/Base_Images_Resized"
    cropped_img_dir = "../data/Cropped_Images"
    prep_img_dir = "../data_prep"

    svm_features_dir = "svm_features/"

    plots_dir = "plots/"
    plots_extension = ".png"

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

    last_layer_before_classifier_name = 'global_average_pooling2d_1'

    svm_train_features_path = svm_features_dir + 'train_features'
    svm_train_labels_path = svm_features_dir + 'train_labels'
    svm_test_features_path = svm_features_dir + 'test_features'
    svm_test_labels_path = svm_features_dir + 'test_labels'
    svm_labels_path = svm_features_dir + 'labels'

    svm_top_n_values = [1, 5]
    svm_cross_validation_sets = 3
    svm_c = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    svm_c_powers = [-3, -2, -1, 0, 1, 2, 3]
    svm_c_points_to_check = 10

    gamma = 'scale'
    square_degree = 2
    svm_probability = True

    svm_kernel_types = ['linear', 'square', 'exp']

    linear_kernel = {'kernel': 'linear', 'probability': svm_probability, 'gamma': gamma}
    square_kernel = {'kernel': 'poly', 'degree': square_degree, 'probability': svm_probability, 'gamma': gamma}
    exp_kernel = {'kernel': 'rbf', 'probability': svm_probability, 'gamma': gamma}

    svm_tuned_parameters = [{'kernel': ['poly'], 'degree': [square_degree], 'gamma': [gamma], 'C': svm_c, 'probability': [svm_probability]}, #kwadratowa
                            {'kernel': ['rbf'], 'gamma': [gamma], 'C': svm_c, 'probability': [svm_probability]}, #exp
                            {'kernel': ['linear'], 'gamma': [gamma], 'C': svm_c, 'probability': [svm_probability]}] #liniowa

    svm_score_types = ['accuracy']