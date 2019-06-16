import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from app_params import AppParams


def draw_learning_history(history, title='Dokładność klasyfikacji procesie uczenia', out_filename=None):
    epochs = np.arange(1, len(history.epoch)+1)
    plt.plot(epochs, history.history['categorical_accuracy'])
    plt.plot(epochs, history.history['val_categorical_accuracy'])
    plt.title(title)
    plt.ylabel('dokładność')
    plt.xlabel('numer epoki')
    plt.legend(['dane trenujące', 'dane testowe'])
    plt.grid()
    if out_filename is not None:
        plt.savefig(AppParams.plots_dir + out_filename + "_cat_acc" + AppParams.plots_extension)
        plt.cla()
        plt.clf()
        plt.close()
    else:
        plt.show()


def draw_learning_history_top_k(history, title='Dokładność klasyfikacji top {AppParams.top_k} w procesie uczenia',
                                out_filename=None):
    epochs = np.arange(1, len(history.epoch)+1)
    plt.plot(epochs, history.history['top_k_categorical_accuracy_metric'])
    plt.plot(epochs, history.history['val_top_k_categorical_accuracy_metric'])
    plt.title(title)
    plt.ylabel('dokładność')
    plt.xlabel('numer epoki')
    plt.legend(['dane trenujące', 'dane testowe'])
    plt.grid()
    if out_filename is not None:
        plt.savefig(AppParams.plots_dir + out_filename + "_top_k_acc" + AppParams.plots_extension)
        plt.cla()
        plt.clf()
        plt.close()
    else:
        plt.show()


def draw_used_params_values_comparision(histories, values, out_filename=None):
    epochs = np.arange(1, len(histories[0].epoch) + 1)
    for history in histories:
        plt.plot(epochs, history.history['val_categorical_accuracy'])
    plt.title('Dokładność klasyfikacji na zbiorze walidującym')
    plt.ylabel('dokładność')
    plt.xlabel('numer epoki')
    plt.legend(values)
    if out_filename is not None:
        plt.savefig(AppParams.plots_dir + out_filename + "_cat_acc" + AppParams.plots_extension)
        plt.cla()
        plt.clf()
        plt.close()
    else:
        plt.show()

    for history in histories:
        plt.plot(epochs, history.history['val_top_k_categorical_accuracy_metric'])
    plt.title(f'Dokładność klasyfikacji top {AppParams.top_k} na zbiorze walidującym')
    plt.ylabel('dokładność')
    plt.xlabel('numer epoki')
    plt.legend(values)
    if out_filename is not None:
        plt.savefig(AppParams.plots_dir + out_filename + "_top_k_acc" + AppParams.plots_extension)
        plt.cla()
        plt.clf()
        plt.close()
    else:
        plt.show()


def draw_two_used_params_values_comparision(histories_dict, out_filename=None):
    legend = []
    params_values = list(histories_dict.keys())
    epochs = np.arange(1, len(histories_dict[params_values[0]].epoch) + 1)
    font_p = FontProperties()
    font_p.set_size('small')
    for params_value_pair in params_values:
        history = histories_dict[params_value_pair]
        plt.plot(epochs, history.history['val_categorical_accuracy'])
    plt.title('Dokładność klasyfikacji na zbiorze walidującym')
    plt.ylabel('dokładność ')
    plt.xlabel('numer epoki')
    plt.legend(params_values, loc='best', prop=font_p)
    if out_filename is not None:
        plt.savefig(AppParams.plots_dir + out_filename + "_cat_acc" + AppParams.plots_extension)
        plt.cla()
        plt.clf()
        plt.close()
    else:
        plt.show()

    for params_value_pair in params_values:
        history = histories_dict[params_value_pair]
        plt.plot(epochs, history.history['val_top_k_categorical_accuracy_metric'])
    plt.title(f'Dokładność klasyfikacji top {AppParams.top_k} na zbiorze walidującym')
    plt.ylabel('dokładność')
    plt.xlabel('numer epoki')
    plt.legend(params_values, loc='best', prop=font_p)
    if out_filename is not None:
        plt.savefig(AppParams.plots_dir + out_filename + "_top_k_acc" + AppParams.plots_extension)
        plt.cla()
        plt.clf()
        plt.close()
    else:
        plt.show()


def plot_validation_curve(svm_type, tested_param, param_values, train_scores_mean, train_scores_std, validation_scores_mean, validation_scores_std):
    plt.title("Krzywa walidacji dla " + svm_type + " SVM")
    plt.xlabel(tested_param)
    plt.ylabel("Dokładność (top-1)")
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
    plt.clf()
    plt.close()


def plot_confusion_matrix(true_labels, predicted_labels, labels,
                          normalize=False, subtitle=None, plot_idx=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        title = 'Znormalizowana macierz pomyłek'
        if subtitle is not None:
            title += subtitle
    else:
        title = 'Macierz pomyłek bez normalizacji'
        if subtitle is not None:
            title += subtitle

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
    plt.clf()
    plt.close()

def plot_top_n_accuracies_for_svm_type(top_n_accs_dict, svm_type, params_c):
    title = "Dokładności"
    for top_n in AppParams.svm_top_n_values:
        title += (" Top-" + str(top_n) + " ")
    title += " - " + " ".join((svm_type, "SVM"))
    plt.title(title)
    plt.xlabel("C")
    plt.ylabel("Dokładności")
    plt.ylim(0.0, 1.1)
    lw = 2

    for top_n, accuracy in top_n_accs_dict.items():
        plt.semilogx(params_c, accuracy, label=("Top-" + str(top_n)), lw=lw)
    plt.legend(loc="best")
    plt.savefig(AppParams.plots_dir + "top_n_accs_" + svm_type + AppParams.plots_extension)
    plt.cla()
    plt.clf()
    plt.close()
    
def plot_roc_curves_for_classes(fpr, tpr, roc_auc, labels, ct_on_plot, subtitle, plot_idx):

    make_equal_parts = lambda lst, part_size: [lst[i:i + part_size] for i in range(0, len(lst), part_size)]
    lw=2
    parts = make_equal_parts(range(0, len(labels)), ct_on_plot)
    j=0
    for p in parts:
        j += 1
        # Plot all ROC curves
        plt.figure()
        for i in range(len(p)):
            plt.plot(fpr[p[i]], tpr[p[i]], lw=lw,
                     label='Krzywa ROC klasy: {0} (area = {1:0.2f})'
                           ''.format(labels[p[i]], roc_auc[p[i]]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Krywa ROC - ' + subtitle)
        plt.legend(loc="best")
        plt.savefig(AppParams.plots_dir + "roc_curve_" + plot_idx + '_' + str(j) + AppParams.plots_extension)
        plt.cla()
        plt.clf()
        plt.close()