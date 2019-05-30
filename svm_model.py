import datetime

import pandas as pd
from keras import Model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from app_params import AppParams


def train_svm(model, modelCreator):
    print(str(datetime.datetime.now()) + ' - start removing classifier')
    model = remove_classifier(model)
    model.summary()
    print(str(datetime.datetime.now()) + ' - start train features extracting')
    #gets images features after cnn - size: (images_cnt, 1280)
    train_features = extract_features(model, modelCreator.train)

    print(str(datetime.datetime.now()) + ' - start test features extracting')
    test_features = extract_features(model, modelCreator.test)

    print(str(datetime.datetime.now()) + ' - start SVM gridsearch')
    do_gridsearch(train_features, modelCreator.train_lab, test_features, modelCreator.test_lab, modelCreator.labels)
    print(str(datetime.datetime.now()) + ' - SVM gridsearch: DONE!')

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
        print(classification_report(y_true=test_labels, y_pred=labels_predicted, output_dict=False))
        print()
        print("Accuracy: {}%".format(svm.score(test_features, test_labels) * 100))
        print("Confusion matrix")
        print(pd.DataFrame(confusion_matrix(test_labels, labels_predicted)))

def remove_classifier(model):
    model.layers.pop()
    model = Model(inputs=model.input, outputs=model.get_layer(AppParams.last_layer_before_classifier_name).output)
    return model

def extract_features(model, images):
    return model.predict(images)