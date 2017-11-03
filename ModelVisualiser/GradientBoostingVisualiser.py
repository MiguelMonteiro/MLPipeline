from IPython.display import display
import pandas as pd
import numpy as np


class GradientBoostingVisualiser(object):
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

        importance = pd.DataFrame(data=np.expand_dims(self.model.feature_importances_, axis=0), columns=self.feature_names)
        importance = importance.transpose()
        self.importance = importance.reindex(importance[0].abs().sort_values(ascending=False).index)

    def display(self):
        n_features = len(self.importance)
        n_features_used = np.count_nonzero(self.importance)
        print('{0}\{1} features used ({2:.1f}%)'.format(n_features_used, n_features,
                                                       n_features_used/float(n_features) * 100))
        pd.set_option('display.max_rows', len(self.importance))
        display(self.importance)
        pd.reset_option('display.max_rows')
