"""
Helper Functions
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


inverse_mapping = {0: 'Very Happy', 1: 'Pretty Happy', 2: 'Not Happy'}

def create_submission_file(filename, ids, preds):
    submission_df = pd.DataFrame({'ID': ids, 'Happy': preds})
    submission_df.to_csv(filename, index=False)

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


def stacked_blending(train, y, test):
	X = train
	y = y
	X_submission = test

	skf = list(StratifiedKFold(y, 2))

	clfs = [RandomForestClassifier(n_estimators=300, n_jobs=-1),
			xgb.XGBClassifier(objective='multi:softmax', learning_rate=.003, subsample=0.7,
                              colsample_bytree=0.7, min_child_weight=30, max_depth=6)]


	print 'Creating train and test sets for blending.'

	dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
	dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

	for j, clf in enumerate(clfs):
		print j, clf
		dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
		for i, (train, test) in enumerate(skf):
			print "Fold", i
			print train
			X_train = X[train]
			y_train = y[train]
			X_test = X[test]
			y_test = y[test]
			clf.fit(X_train, y_train)
			y_submission = clf.predict_proba(X_test)[:,1]
			dataset_blend_train[test, j] = y_submission
			dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
		dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

	print
	print "Blending."
	clf = LogisticRegression(class_weight='auto')
	clf.fit(dataset_blend_train, y)
	y_submission = clf.predict_proba(dataset_blend_test)[:,1]

	print "Linear stretch of predictions to [0,1]"
	y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

	return y_submission
