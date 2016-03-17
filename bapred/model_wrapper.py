from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, LinearSVC

from bapred.utils import binarize_score

# model_wrapper.py is for consistent interface of different models

def sort_features(feature_names, score):
    '''sort features according to their scores'''
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
    
    @staticmethod
    def prec_at_1(model, X, y, n_ans=10):
        Y = model.predict_proba(X)[:, 1]
        return prec_at_1(Y, y, n_ans)


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
    
    @staticmethod
    def prec_at_1(model, X, y, n_ans=10):
        Y = model.decision_function(X)
        return prec_at_1(Y, y, n_ans)


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
    
    @staticmethod
    def prec_at_1(model, X, y, n_ans=10):
        Y = model.decision_function(X)
        return prec_at_1(Y, y, n_ans)


class LinearRegressionWrapper(LinearRegression):
    def predict_score(self, X):
        return self.predict(X)

    def sort_features(self, feature_names):
        return sort_features(feature_names, self.coef_)
    
    @staticmethod
    def prec_at_1(model, X, y, n_ans=10):
        Y = model.predict(X)
        return prec_at_1(Y, y, n_ans)


class RidgeWrapper(Ridge):
    def predict_score(self, X):
        return self.predict(X)

    def sort_features(self, feature_names):
        return sort_features(feature_names, self.coef_)
    
    @staticmethod
    def prec_at_1(model, X, y, n_ans=10):
        Y = model.predict(X)
        return prec_at_1(Y, y, n_ans)