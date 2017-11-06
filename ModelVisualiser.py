from IPython.display import display, Image
import pandas as pd
import numpy as np
import pydotplus
from sklearn import tree


class DecisionTreeClassifierVisualiser(object):
    def __init__(self, model, feature_names, class_names):
        self.model = model
        self.class_names = class_names
        self.feature_names = feature_names

        self.dot_data = tree.export_graphviz(self.model, out_file=None,
                                             feature_names=self.feature_names,
                                             class_names=self.class_names,
                                             filled=True, rounded=True,
                                             special_characters=True)

    def display(self):
        graph = pydotplus.graph_from_dot_data(self.dot_data)
        graph.set_size(13)
        display(Image(graph.create_png()))


class LogisticRegressionVisualiser(object):
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

        weights = pd.DataFrame(data=self.model.coef_, columns=self.feature_names)
        weights = weights.transpose()
        sorted_weights = weights.reindex(weights[0].abs().sort_values(ascending=False).index)

        self.sorted_weights = sorted_weights

    def display(self, max_rows=None):
        n_features = len(self.sorted_weights)
        n_features_used = np.count_nonzero(self.sorted_weights)
        print('{0}\{1} features used ({2:.1f}%)'.format(n_features_used, n_features,
                                                       n_features_used/float(n_features) * 100))
        if max_rows is None:
            max_rows = len(self.sorted_weights)
        # pd.set_option('display.max_rows', max_rows)
        display(self.sorted_weights.iloc[0:max_rows])
        # pd.reset_option('display.max_rows')


class TreeEnsembleVisualiser(object):
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

        importance = pd.DataFrame(data=np.expand_dims(self.model.feature_importances_, axis=0), columns=self.feature_names)
        importance = importance.transpose()
        self.importance = importance.reindex(importance[0].abs().sort_values(ascending=False).index)

    def display(self, max_rows=None):
        n_features = len(self.importance)
        n_features_used = np.count_nonzero(self.importance)
        print('{0}\{1} features used ({2:.1f}%)'.format(n_features_used, n_features,
                                                        n_features_used/float(n_features) * 100))
        if max_rows is None:
            max_rows = len(self.importance)
        display(self.importance.iloc[0:max_rows])





