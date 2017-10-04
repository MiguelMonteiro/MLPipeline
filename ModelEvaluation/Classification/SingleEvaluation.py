from sklearn import metrics
import numpy as np
from .ReportPlots import plot_confusion_matrix, CurvePlot
from .Curves import get_roc_curve, get_precision_recall_curve, get_roc_curve_dict, get_precision_recall_curve_dict
from .EvaluationBaseClasses import BinaryEvaluation, MultiClassEvaluation


def np_array_to_string(array, decimal_places):
    tmp = ['{:.{prec}f}'.format(el, prec=decimal_places) for el in array.flat]
    return np.char.array(tmp).reshape(array.shape)


class SingleEvaluation(object):
    def __init__(self, y_true, y_pred, class_names):
        self.class_names = class_names
        self.accuracy = metrics.accuracy_score(y_true, y_pred)
        self.precision, self.recall, self.f1score, self.support = metrics.precision_recall_fscore_support(y_true,
                                                                                                          y_pred)
        self.confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        self.CurvePlot = CurvePlot

    def plot_confusion_matrix(self):
        # normalize to row totals
        cm_normalized = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis] * 100

        cm_string = np_array_to_string(self.confusion_matrix, 0) + '\n(' + np_array_to_string(cm_normalized, 1) + '%)'

        plot_confusion_matrix(self.confusion_matrix, cm_string, self.class_names)


class SingleEvaluationBinary(SingleEvaluation, BinaryEvaluation):
    def __init__(self, y_true, y_pred, y_score, class_names):
        super(SingleEvaluationBinary, self).__init__(y_true, y_pred, class_names)
        self.roc_curve = get_roc_curve(y_true, y_score)
        self.precision_recall_curve = get_precision_recall_curve(y_true, y_score)


class SingleEvaluationMultiClass(SingleEvaluation, MultiClassEvaluation):
    def __init__(self, y_true, y_pred, y_score, class_names):
        super(SingleEvaluationMultiClass, self).__init__(y_true, y_pred, class_names)
        self.roc_curve = get_roc_curve_dict(y_true, y_score, class_names)
        self.precision_recall_curve = get_precision_recall_curve_dict(y_true, y_score, class_names)

