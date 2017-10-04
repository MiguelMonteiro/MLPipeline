# coding=utf-8
from itertools import cycle
from .ReportPlots import plot_roc_curve, plot_precision_recall_curve
from abc import ABCMeta


class Evaluation(object):

    def to_string(self, var, decimal_places):
        format_spec = '.' + str(decimal_places) + 'f'
        return var.__format__(format_spec)

    def accuracy_to_string(self, decimal_places):
        return self.to_string(self.accuracy, decimal_places)

    def precision_to_string(self, decimal_places):
        return [self.to_string(var, decimal_places) for var in self.precision]

    def recall_to_string(self, decimal_places):
        return [self.to_string(var, decimal_places) for var in self.recall]

    def f1score_to_string(self, decimal_places):
        return [self.to_string(var, decimal_places) for var in self.f1score]

    def support_to_string(self, decimal_places):
        return [self.to_string(var, decimal_places) for var in self.support]


class BinaryEvaluation(Evaluation):

    def get_plot(self, curve, area_name):
        label = u'{0} = {1}'.format(area_name, curve.area_to_string(3))
        return [self.CurvePlot(curve, label=label, color='darkorange')]

    def plot_roc_curve(self):
        roc_plots = self.get_plot(self.roc_curve, 'AUC')
        plot_roc_curve(roc_plots)

    def plot_precision_recall_curve(self):
        precision_recall_plot = self.get_plot(self.precision_recall_curve, 'area')
        plot_precision_recall_curve(precision_recall_plot)


class MultiClassEvaluation(Evaluation):
    def plot_roc_curve(self):
        plots = self.get_plots(self.roc_curve, 'ROC', 'AUC')
        plot_roc_curve(plots)

    def plot_precision_recall_curve(self):
        plots = self.get_plots(self.precision_recall_curve, 'Precision-Recall', 'area')
        plot_precision_recall_curve(plots)

    def get_plots(self, curve_dict, curve_name, area_name):
        plots = []
        for key, color in zip(['macro', 'micro'], ['navy', 'deeppink']):
            label = u'{0}-average {1} curve ({2} = {3})' ''.format(key, curve_name, area_name, curve_dict[key].area_to_string(3))
            plots.append(self.CurvePlot(curve_dict[key], label=label, lw=4, color=color))

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for key, color in zip(self.class_names, colors):
            label = u'{0} curve for class {1} ({2} = {3})' ''.format(curve_name, key, area_name, curve_dict[key].area_to_string(3))
            plots.append(self.CurvePlot(curve_dict[key], label=label, lw=2, color=color))
        return plots
