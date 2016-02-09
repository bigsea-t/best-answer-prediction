from Utils import *
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle

def main(data_dir):
    n_ans = 5
    X, Y = get_data_bin(data_dir, n_ans)

    Xtr, Ytr, Xte, Yte = split_data(X, Y, n_ans)

    models = {
        'Logistic Regression': LogisticRegression(penalty='l2', fit_intercept='True'),
        'Linear SVM': SVC(kernel='linear', probability=True),
        'RBF SVM': SVC(kernel='rbf', probability=True)
    }

    for name, model in models.items():
        model.fit(Xtr, Ytr)

        dump_model(model, name, data_dir)

        Yprob_te = model.predict_log_proba(Xte)
        Yprob_tr = model.predict_log_proba(Xtr)

        print('-- {} --'.format(name))
        print('training set accuracy:', prec_at_1(Yprob_tr, Ytr, n_ans))
        print('test set accuracy:    ', prec_at_1(Yprob_te, Yte, n_ans))
        print()


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
    main(data_dir)

    print('done')