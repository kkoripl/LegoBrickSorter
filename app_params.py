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

    last_layer_before_classifier_name = 'global_average_pooling2d_1'

    # For na zabawę C(od 0.01,1,10,itd.) i gamma (od 0.001,0.01,0.05,1,2,4,10 do 1000)
    svm_cross_validation_sets = 3
    svm_c = [1]#, 10, 100, 1000]
    err_tolerance = [0.9] #1e-1, 1e-2, 1e-3]
    rbf_gamma = ['scale'] #, 1e-4]
    square_degree = [2]

    svm_tuned_parameters = [{'kernel': ['poly'], 'degree': square_degree, 'tol': err_tolerance, 'C': svm_c}, #kwadratowa
                            {'kernel': ['rbf'], 'gamma': rbf_gamma, 'tol': err_tolerance, 'C': svm_c}, #exp
                            {'kernel': ['linear'], 'tol': err_tolerance, 'C': svm_c}] #liniowa

    svm_score_types = ['accuracy']