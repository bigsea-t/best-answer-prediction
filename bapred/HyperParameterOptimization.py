import pickle
import sys

import scipy as sp
from bapred.Heuristic import *
from sklearn.decomposition import TruncatedSVD

from bapred.ModelWrapper import *

#load datas as in main.py
print('Main...')
data_dir = "E:\Posts/"

n_ans = 10
questions, answers, scores = get_raw_data_score(data_dir, n_ans, n_files=102)
print('Data Loaded.')

Xq, Xa, feature_names = transform_raw_data(questions, answers, remove_tag=False)
print('Data transformed')

Y = reguralize_score(scores, n_ans)

Xqtr, Xatr, Ytr, Xqte, Xate, Yte = split_data(Xq, Xa, Y, n_ans)
print('Data Splitted')

Xtr = Xatr
Xte = Xate

#load necessary models
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import LinearSVC
from operator import itemgetter

#the function to format the random_search output and print
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

#possible parameter range
params_space = {
    "LogisticRegression":{'C': np.arange(0.1, 5.1, 0.1),
         'class_weight': [None],
         'dual': [False],
         'fit_intercept': [True, False],
         'intercept_scaling': [1],
         'max_iter': [100],
         'multi_class': ['ovr'],
         'n_jobs': [-1],
         'penalty': ['l2'],
         'random_state': [None],
         'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
         'tol': [0.0001],
         'verbose': [0],
         'warm_start': [False]},
    "SVC":{'C': np.arange(0.1, 5.1, 0.1),
         'cache_size': [200],
         'class_weight': [None],
         'coef0': [0.0],
         'decision_function_shape': [None, 'ovo', 'ovr'],
         'degree': np.arange(1,10,1),
         'gamma': ['auto'],
         'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
         'max_iter': [-1],
         'probability': [True, False],
         'random_state': [None],
         'shrinking': [True, True],
         'tol': [0.001],
         'verbose': [False]},
    "LinearSVC":{'C': np.arange(0.1, 5.1, 0.1),
         'class_weight': [None],
         'dual': [True],
         'fit_intercept': [True, False],
         'intercept_scaling': [1],
         'loss': ['hinge', 'squared_hinge'],
         'max_iter': [1000],
         'multi_class': ['ovr', 'crammer_singer'],
         'penalty': ['l2'],
         'random_state': [None],
         'tol': [0.0001],
         'verbose': [0]},
    'LinearRegression':{'copy_X': [True], 'fit_intercept': [True, False], 'n_jobs': [-1], 'normalize': [True, False]},
    'Ridge':{'alpha': np.arange(0.1, 2.1, 0.1),
         'copy_X': [True],
         'fit_intercept': [True, False],
         'max_iter': [None],
         'normalize': [True, False],
         'random_state': [None],
         'solver': ['auto', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
         'tol': [0.001]}
}

#test on every model
for M in [LogisticRegression, LinearSVC, LinearRegression, Ridge]:
    m = M()
    search = RandomizedSearchCV(m, param_distributions=params_space[m.__class__.__name__], n_jobs=-1, 
                                scoring=prec_at_1_model, n_iter=50)

    search.fit(Xtr, binarize_score(Ytr,n_ans))
    print(M.__name__+":")
    report(search.grid_scores_, 5)
    for i,j in enumerate(sorted(search.grid_scores_, key=itemgetter(1), reverse=True)[:5]):
        print("Model with rank:", i+1)
        mm = m.__class__(**j.parameters)
        mm.fit(Xtr, binarize_score(Ytr, n_ans))
        print("Train Accuracy:", prec_at_1_model(mm, Xtr, Ytr))
        print("Test Accuracy:", prec_at_1_model(mm, Xte, Yte))