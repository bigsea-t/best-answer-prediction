from Utils import *
from ModelWrapper import *
import sys
import pickle


def load_model(name, data_dir):
    with open('{}{}.pkl'.format(data_dir, name), 'rb') as f:
        model = pickle.load(f)
    return model


def dump_model(model, name, data_dir):
    with open('{}{}.pkl'.format(data_dir, name), 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('only one argument is allowed (data_dir)')
    data_dir = sys.argv[1]

    n_ans = 5
    X, Y, feature_names = get_data_score(data_dir, n_ans)

    Y = reguralize_score(Y, n_ans)

    Xtr, Ytr, Xte, Yte = split_data(X, Y, n_ans)

    models = {
        'Logistic Regression': LogisticRegressionWrapper(n_ans=n_ans, penalty='l2', fit_intercept='True'),
        'Linear Regression': LinearRegressionWrapper(),
        'Linear SVM': LinearSVCWrapper(n_ans=n_ans),
        'Ridge Regression': RidgeWrapper(alpha=2)
    }
    # TODO: add linear SVR, some Boosting's

    # NOTE: Non-Linear SVM is not scalable
    # On top of that, Linear SVM is supposed to be enough for this high dimensional features

    for name, model in models.items():
        model.fit(Xtr, Ytr)

        dump_model(model, name, data_dir)

        Yprob_te = model.predict_score(Xte)
        Yprob_tr = model.predict_score(Xtr)

        sorted_features = model.sort_features(feature_names)

        print('-- {} --'.format(name))
        print('training set accuracy:', prec_at_1(Yprob_tr, Ytr, n_ans))
        print('test set accuracy:    ', prec_at_1(Yprob_te, Yte, n_ans))
        print('high score feature:\n', sorted_features[:10])
        print('low score feature :\n', sorted_features[-10:])
        print()

    print('done')