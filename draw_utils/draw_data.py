import matplotlib.pyplot as plt
import numpy as np


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
