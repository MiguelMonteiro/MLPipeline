from IPython.display import display, Image
import pydotplus
import pandas as pd
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
