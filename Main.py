from Utils import *
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle


def load_model(name, data_dir):
    with open('{}{}.pkl'.format(data_dir, name), 'rb') as f:
        model = pickle.load(f)
    return model


def dump_model(model, name, data_dir):
    with open('{}{}.pkl'.format(data_dir, name), 'wb') as f:
        pickle.dump(model, f)


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
        return self.predict_proba(X)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('only one argument is allowed (data_dir)')
    data_dir = sys.argv[1]

    n_ans = 5
    X, Y, feature_names = get_data_score(data_dir, n_ans)

    Xtr, Ytr, Xte, Yte = split_data(X, Y, n_ans)

    models = {
        'Logistic Regression': LogisticRegressionWrapper(n_ans=n_ans, penalty='l2', fit_intercept='True')#,
        # 'Linear SVM': SVC(kernel='linear', probability=True),
        # 'RBF SVM': SVC(kernel='rbf', probability=True)
    }

    for name, model in models.items():
        model.fit(Xtr, Ytr)

        dump_model(model, name, data_dir)

        Yprob_te = model.predict_score(Xte)
        Yprob_tr = model.predict_score(Xtr)

        sorted_features = sort_features(feature_names, model.coef_[0])

        print('-- {} --'.format(name))
        print('training set accuracy:', prec_at_1(Yprob_tr, Ytr, n_ans))
        print('test set accuracy:    ', prec_at_1(Yprob_te, Yte, n_ans))
        print('high score feature:\n', sorted_features[:10])
        print('low score feature :\n', sorted_features[-10:])
        print()

    print('done')