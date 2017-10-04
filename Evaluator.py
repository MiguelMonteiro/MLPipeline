import numpy as np


def get_pred_and_score(x, model):
    y_pred = model.predict(x)
    # some model classes do not have decision function, others do not have predict_proba
    try:
        y_score = model.decision_function(x)
    except AttributeError:
        y_score = model.predict_proba(x)[:, 1].squeeze()
    return y_pred, y_score


class Evaluator(object):
    def __init__(self, y, class_names=None):
        n_classes = len(np.unique(y))
        assert len(class_names) == n_classes

        if len(class_names) == 2:
            from ModelEvaluation import SingleEvaluationBinary
            from ModelEvaluation import CrossValidationEvaluationBinary
            self.SingleEvaluation = SingleEvaluationBinary
            self.CrossValidationEvaluation = CrossValidationEvaluationBinary

        if len(class_names) > 2:
            from ModelEvaluation import SingleEvaluationMultiClass
            from ModelEvaluation import CrossValidationEvaluationMultiClass
            self.SingleEvaluation = SingleEvaluationMultiClass
            self.CrossValidationEvaluation = CrossValidationEvaluationMultiClass

    def evaluate(self, x, y, model, class_names):
        y_pred, y_score = get_pred_and_score(x, model)
        return self.SingleEvaluation(y, y_pred, y_score, class_names)

    def cross_evaluate(self, array_of_evaluations):
        evaluation = self.CrossValidationEvaluation(array_of_evaluations)
        return evaluation
