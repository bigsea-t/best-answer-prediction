from Utils import *
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('only one argument is allowed (data_dir)')
    data_dir = sys.argv[1]

    n_ans = 5
    X, Y = get_data_bin(data_dir, n_ans)

    Xtr, Ytr, Xte, Yte = split_data(X, Y)

    model = LogisticRegression(penalty='l2', fit_intercept='True')
    model.fit(Xtr, Ytr)

    Ypre_te = model.predict(Xte)
    Ypre_tr = model.predict(Xtr)

    print('training set accuracy:', accuracy_score(Ytr, Ypre_tr))
    print('test set accuracy:    ', accuracy_score(Yte, Ypre_te))


