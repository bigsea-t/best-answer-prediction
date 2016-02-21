from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, LinearSVC

from bapred.Utils import *


def sort_features(feature_names, score):
    return sorted(zip(score, feature_names), reverse=True)


class LogisticRegressionWrapper(LogisticRegression):
    def __init__(self, n_ans, *args, **kwargs):
        super(LogisticRegressionWrapper, self).__init__(*args, **kwargs)
        self.n_ans = n_ans

    def fit(self, X, y, sample_weight=None):
        y = binarize_score(y, self.n_ans)
        super(LogisticRegressionWrapper, self).fit(X, y, sample_weight)

    def predict_score(self, X):
        return self.predict_proba(X)[:, 1]

    def sort_features(self, feature_names):
        return sort_features(feature_names, self.coef_[0])


class SVCWrapper(SVC):
    def __init__(self, n_ans, *args, **kwargs):
        super(SVCWrapper, self).__init__(*args, **kwargs)
        self.n_ans = n_ans

    def fit(self, X, y, sample_weight=None):
        y = binarize_score(y, self.n_ans)
        super(SVCWrapper, self).fit(X, y, sample_weight)

    def predict_score(self, X):
        return self.decision_function(X)

    def sort_features(self, feature_names):
        return sort_features(feature_names, self.coef_[0])


class LinearSVCWrapper(LinearSVC):
    def __init__(self, n_ans, *args, **kwargs):
        super(LinearSVCWrapper, self).__init__(*args, **kwargs)
        self.n_ans = n_ans

    def fit(self, X, y):
        y = binarize_score(y, self.n_ans)
        super(LinearSVCWrapper, self).fit(X, y)

    def predict_score(self, X):
        return self.decision_function(X)

    def sort_features(self, feature_names):
        return sort_features(feature_names, self.coef_[0])


class LinearRegressionWrapper(LinearRegression):
    def predict_score(self, X):
        return self.predict(X)

    def sort_features(self, feature_names):
        return sort_features(feature_names, self.coef_)


class RidgeWrapper(Ridge):
    def predict_score(self, X):
        return self.predict(X)

    def sort_features(self, feature_names):
        return sort_features(feature_names, self.coef_)