from IPython.display import display, HTML
import pandas as pd
import numpy as np

class LogisticRegressionVisualiser(object):
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

        weights = pd.DataFrame(data=self.model.coef_, columns=self.feature_names)
        weights = weights.transpose()
        sorted_weights = weights.reindex(weights[0].abs().sort_values(ascending=False).index)

        self.sorted_weights = sorted_weights

    def display(self):
        n_features = len(self.sorted_weights)
        n_features_used = np.count_nonzero(self.sorted_weights)
        print('{0}\{1} features used ({2:.1f}%)'.format(n_features_used, n_features,
                                                       n_features_used/float(n_features) * 100))
        pd.set_option('display.max_rows', len(self.sorted_weights))
        display(self.sorted_weights)
        pd.reset_option('display.max_rows')






