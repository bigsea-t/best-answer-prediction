import pickle
import sys

import scipy as sp
from Heuristic import *
from sklearn.decomposition import TruncatedSVD

from bapred.ModelWrapper import *


def load_model(name, data_dir):
    with open('{}{}.pkl'.format(data_dir, name), 'rb') as f:
        model = pickle.load(f)
    return model


def dump_model(model, name, data_dir):
    with open('{}{}.pkl'.format(data_dir, name), 'wb') as f:
        pickle.dump(model, f)


def generate_similarity(Xq, Xa, n_ans, svd):
    Zq = svd.transform(Xq)
    Za = svd.transform(Xa)
    A = np.repeat(Zq, n_ans, axis=0)
    B = Za
    simZ = np.sum(A * B, axis=1) / (np.sqrt(np.sum(A * A, axis=1)) * np.sqrt(np.sum(B * B, axis=1)) + np.finfo(float).eps)
    return simZ[:, np.newaxis]

def generate_corr(Xq, Xa, n_ans, svd):
    Zq = svd.transform(Xq)
    Za = svd.transform(Xa)
    A = np.repeat(Zq, n_ans, axis=0)
    B = Za

    corr = []
    for a, b in zip(A,B):
        corr.append(np.outer(a,b).flatten())

    return np.vstack(corr)


if __name__ == '__main__':
    print('Main...')
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
        questions, answers, scores = get_raw_data_score(data_dir, n_ans, n_files=102)
        print('Data Loaded.')

        Xq, Xa, feature_names = transform_raw_data(questions, answers, remove_tag=True)
        feature_names.append('correlation qa')
        print('Data transformed')

        Y = reguralize_score(scores, n_ans)

        Xqtr, Xatr, Ytr, Xqte, Xate, Yte = split_data(Xq, Xa, Y, n_ans)
        print('Data Splitted')

        svd = TruncatedSVD(n_components=30)
        Xqatr = sp.sparse.vstack((Xqtr, Xatr))
        svd.fit(Xqatr)
        print('svd fitted')

        # simZtr = generate_similarity(Xqtr, Xatr, n_ans, svd)
        # simZte = generate_similarity(Xqte, Xate, n_ans, svd)
        print('Xqatr', Xqatr.shape)
        simZtr = generate_corr(Xqtr, Xatr, n_ans, svd)
        simZte = generate_corr(Xqte, Xate, n_ans, svd)
        print('simZtr.shape', simZtr.shape)
        print('Sim generated')

        models = {
            'Logistic Regression': LogisticRegressionWrapper(n_ans=n_ans, penalty='l2', fit_intercept='True'),
            # 'Linear Regression': LinearRegressionWrapper(),
            # 'Linear SVM': LinearSVCWrapper(n_ans=n_ans),
            # 'Ridge Regression': RidgeWrapper(alpha=2)
        }
        # TODO: add linear SVR, some Boosting's
    
        # NOTE: Non-Linear SVM is not scalable
        # On top of that, Linear SVM is supposed to be enough for this high dimensional features

        use_cach = False

        print(Xatr.shape)
        Xtr = sp.sparse.hstack((Xatr, simZtr))
        print(Xtr.shape)
        Xte = sp.sparse.hstack((Xate, simZte))

        for name, model in models.items():

            # Xtr = Xatr
            # Xte = Xate

            if use_cach:
                model = load_model(name, data_dir)
            else:
                model.fit(Xtr, Ytr)

                dump_model(model, name, data_dir)
    
            score_ans_tr = model.predict_score(Xtr)
            score_ans_te = model.predict_score(Xte)

            sorted_features = model.sort_features(feature_names)

            print('-- {} --'.format(name))
            print('training set accuracy:', prec_at_1(score_ans_tr, Ytr, n_ans))
            print('test set accuracy:    ', prec_at_1(score_ans_te, Yte, n_ans))
            print('high score feature:\n', sorted_features[:20])
            print('low score feature :\n', sorted_features[-20:])
            print()

    print('done')