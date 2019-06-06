import datetime
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from keras import Model
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.svm import SVC

from app_params import AppParams
from draw_utils import draw_data as dd
from draw_utils.draw_data import plot_confusion_matrix


def train_svm(model, model_creator):
    print(str(datetime.datetime.now()) + ' - start removing classifier')
    model = remove_classifier(model)

    # if not os.path.isfile(AppParams.svm_train_features_path) or not os.path.isfile(
    # AppParams.svm_test_features_path) or not os.path.isfile(AppParams.svm_test_labels_path) or not os.path.isfile(
    # AppParams.svm_labels_path) or not os.path.isfile(AppParams.svm_train_labels_path):
    model_creator.read_prepared_data(binarize_labels=False, save_labels_for_svm=True)

    if os.path.isfile(AppParams.svm_train_features_path+ ".npy"):
        print(str(datetime.datetime.now()) + ' - start train features loading')
        train_features = np.load(AppParams.svm_train_features_path + ".npy")
    else:
        print(str(datetime.datetime.now()) + ' - start train features extracting')
        # gets images features after cnn - size: (images_cnt, 1280)
        train_features = extract_features(model, model_creator.train[0:500])
        np.save(AppParams.svm_train_features_path, train_features)

    if os.path.isfile(AppParams.svm_test_features_path+ ".npy"):
        print(str(datetime.datetime.now()) + ' - start test features loading')
        test_features = np.load(AppParams.svm_test_features_path + ".npy")
    else:
        print(str(datetime.datetime.now()) + ' - start test features extracting')
        test_features = extract_features(model, model_creator.test[0:500])
        np.save(AppParams.svm_test_features_path, test_features)

    # labels = np.load(AppParams.svm_labels_path + ".npy")
    # test_labels = np.load(AppParams.svm_test_labels_path + ".npy")
    # train_labels = np.load(AppParams.svm_train_labels_path + ".npy")

    print(str(datetime.datetime.now()) + ' - start SVM experiments')
    svm_experiments(train_features, model_creator.train_lab[0:500], test_features, model_creator.test_lab[0:500],
                    model_creator.labels)
    print(str(datetime.datetime.now()) + ' - SVM experiments: DONE!')


def remove_classifier(model):
    model.layers.pop()
    model = Model(inputs=model.input, outputs=model.get_layer(AppParams.last_layer_before_classifier_name).output)
    return model


def extract_features(model, images):
    return model.predict(images)


def svm_experiments(train_features, train_labels, test_features, test_labels, labels):
    # check_error_tolerance_influence(train_features, train_labels)
    check_error_tolerance_for_top_n_accuracy(train_features, train_labels, test_features, test_labels, labels)
    # do_gridsearch(train_features, train_labels, test_features, test_labels, labels)


def do_gridsearch(train_features, train_labels, test_features, test_labels, all_labels):
    for score in AppParams.svm_score_types:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        svm = GridSearchCV(SVC(), AppParams.svm_tuned_parameters, scoring=score, cv=AppParams.svm_cross_validation_sets)
        svm.fit(train_features, train_labels)

        print("Best parameters set found on development set:")
        print()
        print(svm.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = svm.cv_results_['mean_test_score']
        stds = svm.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, svm.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        print("SVM CV RESULTS:")
        print()
        print(pd.DataFrame.from_dict(svm.cv_results_))
        print()
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        labels_predicted = svm.predict(test_features)
        plot_confusion_matrix(test_labels, labels_predicted, all_labels, True)
        # print(classification_report(y_true=test_labels, y_pred=labels_predicted, output_dict=False))
        # print()
        # print("Accuracy: {}%".format(svm.score(test_features, test_labels) * 100))
        # print("Confusion matrix")
        # print(pd.DataFrame(confusion_matrix(test_labels, labels_predicted)))
        # class_probs = count_class_probabilities(svm, test_features)
        # acc = count_top_n_accuracies(AppParams.svm_top_n_values, class_probs, test_labels)
        pass


def check_error_tolerance_influence(train_data, train_labels):
    svms = make_svms_dict()
    for svm_type, svm in svms.items():
        print(str(datetime.datetime.now()) + ' - SVM start checking err influence: ' + svm_type)
        measure_validation_stats(svm_type, svm, 'C',
                                 np.logspace(AppParams.svm_c_powers[0], AppParams.svm_c_powers[-1],
                                             AppParams.svm_c_points_to_check),
                                 'accuracy', train_data, train_labels)


def measure_validation_stats(svm_type, svm, tested_param, param_values, scorer, train_data, train_labels):
    train_scores, validation_scores = validation_curve(estimator=svm, X=train_data, y=train_labels,
                                                       param_name=tested_param, param_range=param_values,
                                                       scoring=scorer, cv=AppParams.svm_cross_validation_sets)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)
    dd.plot_validation_curve(svm_type, tested_param, param_values, train_scores_mean, train_scores_std,
                             validation_scores_mean, validation_scores_std)


def check_error_tolerance_for_top_n_accuracy(train_data, train_labels, test_data, test_labels, labels):
    accs_dict = {svm_type:
                    {param_c: None for param_c in AppParams.svm_c}
                for svm_type in AppParams.svm_kernel_types}

    for param_c in AppParams.svm_c:
        svms = make_svms_dict(param_c)
        for svm_type, svm in svms.items():
            print(str(datetime.datetime.now()) + ' - start accuracies for' + svm_type + " - c:" + str(param_c))
            svm.fit(train_data, train_labels)
            labels_probs = svm.predict_proba(test_data)
            labels_predicted = np.argsort(labels_probs, axis=1)[:, -1:]
            accs_dict[svm_type][param_c] = count_top_n_accuracies(AppParams.svm_top_n_values, labels_probs, test_labels)
            plot_confusion_matrix(test_labels, labels_predicted, labels, False, svm_type, param_c,  None, (svm_type + "_c_" + str(param_c)))

    plot_top_n_accuracies(accs_dict)


def plot_top_n_accuracies(svm_c_accuracies):
    accs_dict = defaultdict(list)
    print(str(datetime.datetime.now()) + ' - start ploting top n accuracies')
    for svm_type, c_param_accs in svm_c_accuracies.items():
        for param_c, accs in c_param_accs.items():
            for top_n in AppParams.svm_top_n_values:
                accs_dict[top_n].append(accs[top_n])
        dd.plot_top_n_accuracies_for_svm_type(accs_dict, svm_type, AppParams.svm_c)
        accs_dict.clear()


def make_svms_dict(param_c=None):
    return {'linear': make_linear_kernel_svm(param_c),
            'square': make_square_kernel_svm(param_c),
            'exp': make_exp_kernel_svm(param_c)}


def make_linear_kernel_svm(param_c=None):
    if param_c is not None:
        return SVC(kernel=AppParams.linear_kernel['kernel'],
                   probability=AppParams.linear_kernel['probability'],
                   gamma=AppParams.linear_kernel['gamma'],
                   C=param_c)
    else:
        return SVC(kernel=AppParams.linear_kernel['kernel'],
                   probability=AppParams.linear_kernel['probability'],
                   gamma=AppParams.linear_kernel['gamma'])


def make_square_kernel_svm(param_c=None):
    if param_c is not None:
        return SVC(kernel=AppParams.square_kernel['kernel'],
                   degree=AppParams.square_kernel['degree'],
                   probability=AppParams.square_kernel['probability'],
                   gamma=AppParams.square_kernel['gamma'],
                   C=param_c)
    else:
        return SVC(kernel=AppParams.square_kernel['kernel'],
                   degree=AppParams.square_kernel['degree'],
                   probability=AppParams.square_kernel['probability'],
                   gamma=AppParams.square_kernel['gamma'])


def make_exp_kernel_svm(param_c=None):
    if param_c is not None:
        return SVC(kernel=AppParams.exp_kernel['kernel'],
                   probability=AppParams.exp_kernel['probability'],
                   gamma=AppParams.exp_kernel['gamma'],
                   C=param_c)
    else:
        return SVC(kernel=AppParams.exp_kernel['kernel'],
                   probability=AppParams.exp_kernel['probability'],
                   gamma=AppParams.exp_kernel['gamma'])


def count_class_probabilities(svm, data):
    return svm.predict_proba(data)


def count_top_n_accuracies(n_values, class_probs, true_labels):
    return {n: count_top_n_accuracy(n, class_probs, true_labels) for n in n_values}


def count_top_n_accuracy(n, class_probs, true_labels):
    topNclasses = np.argsort(class_probs, axis=1)[:, -n:]
    return np.mean(np.array([1 if true_labels[i] in topNclasses[i] else 0 for i in range(len(topNclasses))]))
