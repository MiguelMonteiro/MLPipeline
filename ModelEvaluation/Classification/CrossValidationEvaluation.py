import numpy as np
from .RandomVariable import RandomVariable
from .ReportPlots import plot_confusion_matrix, CurvePlotCrossValidation
from .Curves import random_variable_from_array_of_curves
from .EvaluationBaseClasses import BinaryEvaluation, MultiClassEvaluation


class CrossValidationEvaluation(object):
    def __init__(self, array_of_evaluations):

        self.class_names = array_of_evaluations[0].class_names
        accuracy, precision, recall, f1score, support, confusion_matrix = [], [], [], [], [], []

        for evaluation in array_of_evaluations:
            accuracy.append(evaluation.accuracy)
            precision.append(evaluation.precision)
            recall.append(evaluation.recall)
            f1score.append(evaluation.f1score)
            support.append(evaluation.support)
            confusion_matrix.append(evaluation.confusion_matrix)

        self.accuracy = RandomVariable(accuracy)
        self.precision = [RandomVariable(r) for r in np.array(precision).transpose()]
        self.recall = [RandomVariable(r) for r in np.array(recall).transpose()]
        self.f1score = [RandomVariable(r) for r in np.array(f1score).transpose()]
        self.support = [RandomVariable(r) for r in np.array(support).transpose()]
        self.confusion_matrix = RandomVariable(confusion_matrix)

        self.CurvePlot = CurvePlotCrossValidation

    def plot_confusion_matrix(self):

        normalized_cm = RandomVariable(np.apply_along_axis(func1d=lambda x: x.astype('float')/x.sum(),
                                                           axis=2, arr=self.confusion_matrix.array) * 100)

        cm_string = self.confusion_matrix.to_string(0) + '\n(' + normalized_cm.to_string(1) + '%)'

        plot_confusion_matrix(self.confusion_matrix.mean, cm_string, self.class_names)


class CrossValidationEvaluationBinary(CrossValidationEvaluation, BinaryEvaluation):
    def __init__(self, array_of_evaluations):
        super(CrossValidationEvaluationBinary, self).__init__(array_of_evaluations)

        self.roc_curve = random_variable_from_array_of_curves(
            [evaluation.roc_curve for evaluation in array_of_evaluations])
        self.precision_recall_curve = random_variable_from_array_of_curves(
            [evaluation.precision_recall_curve for evaluation in array_of_evaluations])


class CrossValidationEvaluationMultiClass(CrossValidationEvaluation, MultiClassEvaluation):
    def __init__(self, array_of_evaluations):
        super(CrossValidationEvaluationMultiClass, self).__init__(array_of_evaluations)
        keys = self.class_names + ['micro', 'macro']
        self.roc_curve = {}
        self.precision_recall_curve = {}
        for key in keys:
            self.roc_curve[key] = random_variable_from_array_of_curves(
                [evaluation.roc_curve[key] for evaluation in array_of_evaluations])
            self.precision_recall_curve[key] = random_variable_from_array_of_curves(
                [evaluation.precision_recall_curve[key] for evaluation in array_of_evaluations])



