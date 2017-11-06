import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import interp
from .RandomVariable import RandomVariable


class Curve(object):
    def __init__(self, x, y, thresholds, area):
        self.x = x
        self.y = y
        self.thresholds = thresholds
        self.area = area

    def area_to_string(self, decimal_places):
        format_spec = '.' + str(decimal_places) + 'f'
        return self.area.__format__(format_spec)


class ROCCurve(Curve):
    def __init__(self, fpr, tpr, thresholds, auc):
        super(ROCCurve, self).__init__(fpr, tpr, thresholds, auc)

    @property
    def fpr(self):
        return self.x

    @property
    def tpr(self):
        return self.y

    @property
    def auc(self):
        return self.area


class PrecisionRecallCurve(Curve):
    def __init__(self, recall, precision, thresholds, average_precision_score):
        super(PrecisionRecallCurve, self).__init__(recall, precision, thresholds, average_precision_score)

    @property
    def recall(self):
        return self.x

    @property
    def precision(self):
        return self.y

    @property
    def average_precision_score(self):
        return self.area


def get_roc_curve(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    return ROCCurve(fpr, tpr, thresholds, auc)


def get_precision_recall_curve(y_true, y_score):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    # to match roc curve code
    thresholds = np.concatenate((thresholds, [np.max(y_score) + 1]))
    average_precision_score = metrics.average_precision_score(y_true, y_score)
    return PrecisionRecallCurve(recall, precision, thresholds, average_precision_score)


def macro_average_curve(curve_dict, n_classes):
    # Compute macro-average ROC curve
    # First aggregate all false positive rates
    all_x = np.unique(np.concatenate([item.x for item in curve_dict.itervalues()]))

    # Then interpolate all ROC curves at this points
    mean_y = np.zeros_like(all_x)
    mean_thresholds = np.zeros_like(all_x)
    for item in curve_dict.itervalues():
        mean_y += interp(all_x, item.x, item.y)
        mean_thresholds += interp(all_x, item.x, item.thresholds)
    # Finally average it and compute AUC
    mean_y /= n_classes
    mean_thresholds /= n_classes

    return all_x, mean_y, mean_thresholds


def get_roc_curve_dict(y_true, y_score, class_names):
    # Compute ROC curve and ROC area for each class
    y_true = np.array(pd.get_dummies(y_true))
    n_classes = len(class_names)
    curve_dict = {}
    for idx, name in enumerate(class_names):
        curve_dict[name] = get_roc_curve(y_true[:, idx], y_score[:, idx])

    # Compute micro-average ROC curve
    curve_dict['micro'] = get_roc_curve(y_true.ravel(), y_score.ravel())

    # Compute macro-average ROC curve
    all_fpr, mean_tpr, mean_thresholds = macro_average_curve(curve_dict, n_classes)
    auc_macro = metrics.auc(all_fpr, mean_tpr)
    curve_dict['macro'] = ROCCurve(all_fpr, mean_tpr, mean_thresholds, auc_macro)

    return curve_dict


def get_precision_recall_curve_dict(y_true, y_score, class_names):
    # Compute ROC curve and ROC area for each class
    y_true = np.array(pd.get_dummies(y_true))
    n_classes = len(class_names)
    curve_dict = {}
    for idx, name in enumerate(class_names):
        curve_dict[name] = get_precision_recall_curve(y_true[:, idx], y_score[:, idx])

    # Compute micro-average ROC curve and ROC area
    precision, recall, thresholds = metrics.precision_recall_curve(y_true.ravel(), y_score.ravel())
    thresholds = np.concatenate((thresholds, [np.max(y_score) + 1]))
    average_precision_score = metrics.average_precision_score(y_true, y_score, average='micro')
    curve_dict['micro'] = PrecisionRecallCurve(recall, precision, thresholds, average_precision_score)

    # Compute macro-average precision recall curve
    all_recall, mean_precision, mean_thresholds = macro_average_curve(curve_dict, n_classes)

    average_precision_score = metrics.average_precision_score(y_true, y_score, average='macro')
    curve_dict['macro'] = PrecisionRecallCurve(all_recall, mean_precision, mean_thresholds, average_precision_score)

    return curve_dict


def make_cross_validation_roc_curve(array_of_curves):
    base_x = np.linspace(0, 1, 101)

    y = np.array([interp(base_x, curve.x, curve.y) for curve in array_of_curves])
    y[:, 0] = 0
    y = RandomVariable(y)
    # use to have curve.y as second argument
    thresholds = np.array([interp(base_x, curve.x, curve.thresholds) for curve in array_of_curves])
    thresholds = RandomVariable(thresholds)

    area = RandomVariable([curve.area for curve in array_of_curves])
    return ROCCurve(base_x, y, thresholds, area)


def make_cross_validation_precision_recall_curve(array_of_curves):
    base_x = np.linspace(0, 1, 101)

    y = np.array([interp(base_x, np.flip(curve.x, 0), np.flip(curve.y, 0)) for curve in array_of_curves])
    y[:, 0] = 0
    y = RandomVariable(y)
    # use to have curve.y as second argument
    thresholds = np.array([interp(base_x, np.flip(curve.x, 0), np.flip(curve.thresholds, 0))
                           for curve in array_of_curves])
    thresholds = RandomVariable(thresholds)

    area = RandomVariable([curve.area for curve in array_of_curves])

    return PrecisionRecallCurve(base_x, y, thresholds, area)
