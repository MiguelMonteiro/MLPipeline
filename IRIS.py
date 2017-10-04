import numpy as np
from sklearn import datasets, linear_model

from MLPipeline.ModelEvaluation.Classification.Report import long_report
from MLPipeline.Pipelines import OnePassPipeline, KFoldCrossValidationPipeline, LeaveOneOutCrossValidationPipeline

data = datasets.load_iris()

x = np.array(data.data)
y = np.array(data.target)
#
#y[y == 2] = 0

class_names = ['a', 'b', 'c']

model = linear_model.LogisticRegression(penalty='l2', C=1e2)

pipeline = OnePassPipeline(x, y, model, class_names)
evaluation_1 = pipeline.run_pipeline(report=False)
long_report(y, evaluation_1[0], evaluation_1[1])

pipeline = KFoldCrossValidationPipeline(x, y, model, class_names, 6)
evaluation_2 = pipeline.run_pipeline()
long_report(y, evaluation_2[0], evaluation_2[1])

# pipeline = LeaveOneOutCrossValidationPipeline(x, y, model, class_names)
# evaluation_3 = pipeline.run_pipeline()
# long_report(y, evaluation_3[0], evaluation_3[1])

print(' ')
print('Done!')
