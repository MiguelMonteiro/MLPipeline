from sklearn.model_selection import ParameterGrid


class Search(object):
    def __init__(self, parameters, model, train_evaluation, valid_evaluation):
        self.parameters = parameters
        self.model = model
        self.train_evaluation = train_evaluation
        self.valid_evaluation = valid_evaluation


class ModelSearcher(object):
    def __init__(self, pipeline, model, param_grid):
        self.pipeline = pipeline
        self.model = model
        self.param_grid = param_grid
        self.search_list = []

    def search(self, report=False):
        for param in list(ParameterGrid(self.param_grid)):
            if report:
                print param
            model = self.model(**param)
            self.pipeline.model = model
            train_evaluation, valid_evaluation = self.pipeline.run_pipeline(report=False)
            self.search_list.append(Search(param, model, train_evaluation, valid_evaluation))

    def get_all_models_searched_over(self):
        if not self.search_list:
            self.search()
        return self.search_list

    def get_model_with_best_valid_accuracy(self):
        if not self.search_list:
            self.search()

        best = self.search_list[0]
        for element in self.search_list:
            if element.valid_evaluation.accuracy > best.valid_evaluation.accuracy:
                best = element
        return best

    def get_model_with_best_valid_auc(self):
        if not self.search_list:
            self.search()

        best = self.search_list[0]
        for element in self.search_list:
            if element.valid_evaluation.roc_curve.auc > best.valid_evaluation.roc_curve.auc:
                best = element
        return best

    def get_models_sorted_by_decreasing_valid_accuracy(self):
        if not self.search_list:
            self.search()
        return sorted([element for element in self.search_list],
                      key=lambda x: x.valid_evaluation.accuracy, reverse=True)

    def get_models_sorted_by_decreasing_valid_auc(self):
        if not self.search_list:
            self.search()
        return sorted([element for element in self.search_list],
                      key=lambda x: x.valid_evaluation.roc_curve.auc, reverse=True)



