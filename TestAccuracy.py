from Utils import *
from ModelWrapper import *
from Heuristic import *
from sklearn.decomposition import TruncatedSVD
import sys
import pickle
import scipy as sp

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Usage: python Main.py <data_dir>')
    data_dir = sys.argv[1]

    train_acc = [[],[]]
    test_acc = [[],[]]
    for n_ans in range(519):
        questions, answers, scores = get_raw_data_score(data_dir, n_ans)
        if len(questions)==0:
            continue
        for rt in [True, False]:
            Xq, Xa, feature_names = transform_raw_data(questions, answers, rt)
            feature_names.append('correlation qa')

            Y = reguralize_score(scores, n_ans)

            Xqtr, Xatr, Ytr, Xqte, Xate, Yte = split_data(Xq, Xa, Y, n_ans)
            if Xatr.size==0 or Xate.size==0:
                continue

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
                # Xtr = sp.sparse.hstack((Xatr, simZtr[:, np.newaxis]))
                # Xte = sp.sparse.hstack((Xate, simZte[:, np.newaxis]))
                Xtr = Xatr
                Xte = Xate

                model.fit(Xtr, Ytr)

                score_ans_tr = model.predict_score(Xtr)
                score_ans_te = model.predict_score(Xte)

                sorted_features = model.sort_features(feature_names)
                trainacc = prec_at_1(score_ans_tr, Ytr, n_ans)
                testacc = prec_at_1(score_ans_te, Yte, n_ans)
                print('-- {}, {}, {} --'.format(name, n_ans, rt))
                print('training set accuracy:', trainacc)
                print('test set accuracy:    ', testacc)
                print('high score feature:\n', sorted_features[:10])
                print('low score feature :\n', sorted_features[-10:])
                print()
                train_acc[int(rt)].append((n_ans, name, trainacc))
                test_acc[int(rt)].append((n_ans, name, testacc))

            print('done')
    print(repr(train_acc))
    print(repr(test_acc))