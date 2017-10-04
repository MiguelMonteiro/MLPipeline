import itertools
import matplotlib.pyplot as plt
import numpy as np


class CurvePlot(object):
    def __init__(self, curve, label, color, **kwargs):
        self.curve = curve
        self.label = label
        self.color = color
        self.kwargs = kwargs

    def plot(self):
        plt.plot(self.curve.x, self.curve.y, label=self.label, color=self.color, **self.kwargs)


class CurvePlotCrossValidation(object):
    def __init__(self, curve, label, color, **kwargs):
        self.curve = curve
        self.label = label
        self.color = color
        self.kwargs = kwargs

    def plot(self):
        plt.plot(self.curve.x, self.curve.y.mean, label=self.label, color=self.color, **self.kwargs)
        y_upper = np.minimum(self.curve.y.mean + self.curve.y.std, 1)
        y_lower = self.curve.y.mean - self.curve.y.std
        plt.fill_between(self.curve.x, y_lower, y_upper, color=self.color, alpha=.3)


def plot_roc_curve(roc_plots):

    plt.figure()
    for roc_plot in roc_plots:
        roc_plot.plot()

    plt.axis('scaled')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.plot([0, 1], [1, 0], color='grey', lw=.1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show(block=True)


def plot_precision_recall_curve(plots):

    plt.figure()
    for plot in plots:
        plot.plot()

    plt.axis('scaled')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.show(block=True)


def plot_confusion_matrix(cm, cm_string, class_names, cmap=plt.cm.Blues):
    title = 'Confusion matrix'
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 1.2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm_string[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show(block=True)

