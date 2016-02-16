from Utils import *
from ModelWrapper import *
from Heuristic import *
from sklearn.decomposition import TruncatedSVD
import sys
import pickle
import scipy as sp


def load_model(name, data_dir):
    with open('{}{}.pkl'.format(data_dir, name), 'rb') as f:
        model = pickle.load(f)
    return model


def dump_model(model, name, data_dir):
    with open('{}{}.pkl'.format(data_dir, name), 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception('Usage: python Main.py <data_dir> <ML/Heuristic>[0/1]')
    data_dir = sys.argv[1]
    h_mode = int(sys.argv[2])

    n_ans = 5
    if h_mode:
        _, X, Y = get_raw_data_score(data_dir, n_ans)
        Y = reguralize_score(Y, n_ans)
        
        models = {
            'Dummy Heuristic': DummyHeuristic(n_ans),
            'Length Heuristic by Char': CharLengthHeuristic(n_ans),
            'Length Heuristic by Word': WordLengthHeuristic(n_ans),
            'Length Heuristic by Char and Code': CharNCodeLengthHeuristic(n_ans),
            'Length Heuristic by Sentence': SentenceLengthHeuristic(n_ans),
            'Length Heuristic by Sentence and Code': SentenceNCodeLengthHeuristic(n_ans),
            'Length Heuristic by Average Sentence Length': AvgSentenceLengthHeuristic(n_ans)
        }
        
        for name, model in models.items():
            Yprob = model.predict_score(X)
    
            print('-- {} --'.format(name))
            print('accuracy:    ', prec_at_1(Yprob, Y, n_ans))
            print()
            
    else:
        questions, answers, scores = get_raw_data_score(data_dir, n_ans)
        Xq, Xa, feature_names = transform_raw_data(questions, answers)
        feature_names.append('correlation qa')

        Y = reguralize_score(scores, n_ans)

        Xqtr, Xatr, Ytr, Xqte, Xate, Yte = split_data(Xq, Xa, Y, n_ans)

        svd = TruncatedSVD(n_components=100)
        Xqatr = sp.sparse.vstack((Xqtr, Xatr))
        svd.fit(Xqatr)

        Zqtr = svd.transform(Xqtr)
        Zatr = svd.transform(Xatr)
        A = np.repeat(Zqtr, n_ans, axis=0)
        B = Zatr
        simZtr = np.sum(A * B, axis=1) / (np.sqrt(np.sum(A * A, axis=1)) * np.sqrt(np.sum(B * B, axis=1)) + np.finfo(float).eps)

        Zqte = svd.transform(Xqte)
        Zate = svd.transform(Xate)
        simZte = np.sum(np.repeat(Zqte, n_ans, axis=0) * Zate, axis=1)


        models = {
            'Logistic Regression': LogisticRegressionWrapper(n_ans=n_ans, penalty='l2', fit_intercept='True'),
            # 'Linear Regression': LinearRegressionWrapper(),
            # 'Linear SVM': LinearSVCWrapper(n_ans=n_ans),
            # 'Ridge Regression': RidgeWrapper(alpha=2)
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

            dump_model(model, name, data_dir)
    
            score_ans_tr = model.predict_score(Xtr)
            score_ans_te = model.predict_score(Xte)

            sorted_features = model.sort_features(feature_names)

            print('-- {} --'.format(name))
            print('training set accuracy:', prec_at_1(score_ans_tr, Ytr, n_ans))
            print('test set accuracy:    ', prec_at_1(score_ans_te, Yte, n_ans))
            print('high score feature:\n', sorted_features[:10])
            print('low score feature :\n', sorted_features[-10:])
            print()

            model2 = LogisticRegressionWrapper(n_ans=n_ans, penalty='l2', fit_intercept='True')
            Xtr2 = np.hstack((score_ans_tr[:, np.newaxis], simZtr[:, np.newaxis]))
            Xte2 = np.hstack((score_ans_te[:, np.newaxis], simZte[:, np.newaxis]))

            model2.fit(Xtr2, Ytr)

            score_te = model2.predict_score(Xte2)
            score_tr = model2.predict_score(Xtr2)
    
            print('- merge qa correlation -')
            print('training set accuracy:', prec_at_1(score_tr, Ytr, n_ans))
            print('test set accuracy:    ', prec_at_1(score_te, Yte, n_ans))
            print('coef', model2.coef_)
            print()

    print('done')