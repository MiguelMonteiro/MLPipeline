import numpy as np
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from .Evaluator import Evaluator, get_pred_and_score
from .ModelEvaluation.Classification.Report import short_report


class OnePassPipeline(object):
    def __init__(self, x, y, model, class_names, scale=True, test_size=.3):
        self.x = x
        self.y = y
        self.model = model
        self.class_names = class_names
        self.scale = scale
        self.test_size = test_size

    def run_pipeline(self, report=True):

        evaluator = Evaluator(self.y, self.class_names)

        x, y = shuffle(self.x, self.y, random_state=0)

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=.3, random_state=0)

        if self.scale:
            x_train, x_valid = scale(x_train, x_valid)

        self.model.fit(x_train, y_train)

        train_evaluation = evaluator.evaluate(x_train, y_train, self.model, self.class_names)
        valid_evaluation = evaluator.evaluate(x_valid, y_valid, self.model, self.class_names)

        if report:
            short_report(y, train_evaluation, valid_evaluation)

        return train_evaluation, valid_evaluation


def scale(x_train, x_valid):
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)
    return x_train, x_valid


class KFoldCrossValidationPipeline(object):
    def __init__(self, x, y, model, class_names, k_folds, scale=True):
        self.x = x
        self.y = y
        self.model = model
        self.class_names = class_names
        self.k_folds = k_folds
        self.data_generator = KFold(n_splits=k_folds, random_state=0, shuffle=False)
        self.scale = scale

    def run_pipeline(self, report=True):

        evaluator = Evaluator(self.y, self.class_names)

        x, y = shuffle(self.x, self.y, random_state=0)

        train_evaluation_array = []
        valid_evaluation_array = []

        for train_index, valid_index in self.data_generator.split(x):

            x_train = x[train_index]
            x_valid = x[valid_index]

            if self.scale:
                x_train, x_valid = scale(x_train, x_valid)

            self.model.fit(x_train, y[train_index])
            train_evaluation_array.append(evaluator.evaluate(x_train, y[train_index], self.model, self.class_names))
            valid_evaluation_array.append(evaluator.evaluate(x_valid, y[valid_index], self.model, self.class_names))

        train_evaluation = evaluator.cross_evaluate(train_evaluation_array)
        valid_evaluation = evaluator.cross_evaluate(valid_evaluation_array)

        if report:
            short_report(y, train_evaluation, valid_evaluation)

        return train_evaluation, valid_evaluation


class LeaveOneOutCrossValidationPipeline(object):
    def __init__(self, x, y, model, class_names, scale=True):
        self.x = x
        self.y = y
        self.model = model
        self.class_names = class_names
        self.data_generator = LeaveOneOut()
        self.scale = scale

    def run_pipeline(self, report=True):

        evaluator = Evaluator(self.y, self.class_names)

        x, y = shuffle(self.x, self.y, random_state=0)

        train_evaluation_array = []

        y_valid = []
        y_valid_pred = []
        y_valid_score = []

        for train_index, valid_index in self.data_generator.split(x):

            x_train = x[train_index]
            x_valid = x[valid_index]

            if self.scale:
                x_train, x_valid = scale(x_train, x_valid)

            self.model.fit(x_train, y[train_index])

            train_evaluation_array.append(evaluator.evaluate(x_train, y[train_index], self.model, self.class_names))

            y_valid.append(y[valid_index])
            pred, score = get_pred_and_score(x_valid, self.model)
            y_valid_pred.append(pred)
            y_valid_score.append(score)

        train_evaluation = evaluator.cross_evaluate(train_evaluation_array)
        valid_evaluation = evaluator.SingleEvaluation(np.array(y_valid).ravel(), np.array(y_valid_pred).ravel(),
                                                      np.array(y_valid_score).squeeze(), self.class_names)
        if report:
            short_report(y, train_evaluation, valid_evaluation)

        return train_evaluation, valid_evaluation
