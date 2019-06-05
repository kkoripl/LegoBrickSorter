import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from app_params import AppParams


def draw_learning_history(history, title='Dokładność modelu w procesie uczenia'):
    epochs = np.arange(1, len(history.epoch)+1)
    plt.plot(epochs, history.history['categorical_accuracy'])
    plt.plot(epochs, history.history['val_categorical_accuracy'])
    plt.title(title)
    plt.ylabel('dokładność [%]')
    plt.xlabel('numer epoki')
    plt.legend(['dane trenujące', 'dane testowe'])
    plt.grid()
    plt.show()


def plot_validation_curve(svm_type, tested_param, param_values, train_scores_mean, train_scores_std, validation_scores_mean, validation_scores_std):
    plt.title("Krzywa walidacji dla " + svm_type + " SVM")
    plt.xlabel(tested_param)
    plt.ylabel("Dokładność (top-1) [%]")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_values, train_scores_mean, label="Zbiory trenujące",
                 color="darkorange", lw=lw)
    plt.fill_between(param_values, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_values, validation_scores_mean, label="Walidacja krzyżowa",
                 color="navy", lw=lw)
    plt.fill_between(param_values, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig(AppParams.plots_dir + "svm_validation_curve_" + svm_type + AppParams.plots_extension)
    plt.cla()
    plt.close()


def plot_confusion_matrix(true_labels, predicted_labels, labels,
                          normalize=False,
                          title=None,
                          plot_idx=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Znormalizowana macierz pomyłek'
        else:
            title = 'Macierz pomyłek bez normalizacji'

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    # Only use the labels that appear in the data
    labels = [labels[lab_idx] for lab_idx in unique_labels(true_labels, predicted_labels)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='Rzeczywista klasa',
           xlabel='Przewidywana klasa')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(AppParams.plots_dir + "confusion_matrix_" + plot_idx + AppParams.plots_extension)
    plt.cla()
    plt.close()