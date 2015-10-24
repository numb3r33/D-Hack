"""
Helper Functions
"""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

inverse_mapping = {0: 'Very Happy', 1: 'Pretty Happy', 2: 'Not Happy'}

def create_submission_file(filename, ids, preds):
    submission_df = pd.DataFrame({'ID': ids, 'Happy': preds})
    submission_df.to_csv('./submissions/' + filename, index=False)

def inverse_mapping_func(preds):
    return [inverse_mapping[pred] for pred in preds]

def score(y, ypred):
    conf_mat = confusion_matrix(y, ypred)
    cost_mat = np.array([[50, 10, 5], [-5, 50, 10], [-10, -5, 50]])
    cost_mult = np.multiply(conf_mat, cost_mat)
    cost_sum = cost_mult.sum(axis=0)
    cost_sum_total = np.sum(cost_sum)

    return cost_sum_total * 1. / (y.shape[0] * 50)

def score_xgb(ypred, y):
    y = y.get_label()
    conf_mat = confusion_matrix(y, ypred)
    cost_mat = np.array([[50, 10, 5], [-5, 50, 10], [-10, -5, 50]])
    cost_mult = np.multiply(conf_mat, cost_mat)
    cost_sum = cost_mult.sum(axis=0)
    cost_sum_total = np.sum(cost_sum)

    custom_score =  cost_sum_total * 1. / (y.shape[0] * 50)

    return 'custom_score', custom_score

def make_scorer_func():
    return make_scorer(score, greater_is_better=True)

def split_dataset(dfsurvey, train_labels):
    sss = StratifiedShuffleSplit(train_labels, test_size=0.3)
    surveytrain, surveytest = next(iter(sss))
    surveymask = np.ones(dfsurvey.shape[0], dtype='int')
    surveymask[surveytrain] = 1
    surveymask[surveytest] = 0

    surveymask = (surveymask==1)
    return surveymask

def convert_to_labels(train, test, col_name):
    train_test_append = np.hstack([train[col_name], test[col_name]])
    lbl = LabelEncoder()

    lbl.fit(train_test_append)

    train_convert = lbl.transform(train[col_name])
    test_convert = lbl.transform(test[col_name])

    return (train_convert, test_convert)
