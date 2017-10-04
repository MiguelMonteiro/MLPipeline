import numpy as np
decimal_places = 3


def class_balance_report(y, class_names):
    classes, counts = np.unique(y, return_counts=True)
    class_percentages = counts.astype(float) / np.sum(counts) * 100
    print('The class balance is:')
    for c in classes:
        print('Class ' + class_names[c] + ' (' + repr(c) + ')' + ': %.2f%%' % class_percentages[c])


def print_dashed_line():
    print('------------------------------------------------------')


def accuracy_report(train_evaluation, valid_evaluation):
    print('The training accuracy was ' + train_evaluation.accuracy_to_string(decimal_places))
    print('The validation accuracy was ' + valid_evaluation.accuracy_to_string(decimal_places))


def precision_recall_f1score_support_report(evaluation):

    d = {'precision': evaluation.precision_to_string(decimal_places),
         'recall': evaluation.recall_to_string(decimal_places),
         'F1 score': evaluation.f1score_to_string(decimal_places),
         'support': evaluation.support_to_string(decimal_places)}

    for key in d:
        for i in range(len(evaluation.class_names)):
            class_name = evaluation.class_names[i]
            var = d[key][i]
            print('The ' + key + ' for class ' + class_name + ' is ' + var)
        print_dashed_line()


def auc_report_binary(train_evaluation, valid_evaluation):

    print('The AUC of the training set is ' + train_evaluation.roc_curve.area_to_string(decimal_places))
    print('The AUC of the validation set is ' + valid_evaluation.roc_curve.area_to_string(decimal_places))


def auc_report_multiclass(train_evaluation, valid_evaluation):
    train_auc_dict_strings = {key: item.area_to_string(decimal_places) for key, item in
                              train_evaluation.roc_curve.iteritems()}
    valid_auc_dict_strings = {key: item.area_to_string(decimal_places) for key, item in
                              valid_evaluation.roc_curve.iteritems()}
    for class_name in train_evaluation.class_names:
        print('The training AUC for class ' + class_name + ' is ' + train_auc_dict_strings[class_name])
        print('The validation AUC for class ' + class_name + ' is ' + valid_auc_dict_strings[class_name])
        print_dashed_line()
    for key in ['micro', 'macro']:
        print('The ' + key + ' average training AUC for class ' + class_name + ' is ' + train_auc_dict_strings[key])
        print('The ' + key + ' average validation AUC for class ' + class_name + ' is ' + valid_auc_dict_strings[key])
        print_dashed_line()


def auc_report(train_evaluation, valid_evaluation):
    if len(train_evaluation.class_names) > 2:
        return auc_report_multiclass(train_evaluation, valid_evaluation)
    return auc_report_binary(train_evaluation, valid_evaluation)


def long_report(y, train_evaluation, valid_evaluation):
    print_dashed_line()
    print_dashed_line()
    print('EVALUATION:')
    class_balance_report(y, train_evaluation.class_names)
    print_dashed_line()

    accuracy_report(train_evaluation, valid_evaluation)
    print_dashed_line()
    print('For the validation set:')
    precision_recall_f1score_support_report(valid_evaluation)

    auc_report(train_evaluation, valid_evaluation)
    print_dashed_line()
    print_dashed_line()

    valid_evaluation.plot_confusion_matrix()
    valid_evaluation.plot_roc_curve()
    valid_evaluation.plot_precision_recall_curve()


def short_report(y, train_evaluation, valid_evaluation):
    print('EVALUATION:')
    print_dashed_line()
    accuracy_report(train_evaluation, valid_evaluation)
    print_dashed_line()
    auc_report(train_evaluation, valid_evaluation)
    print_dashed_line()