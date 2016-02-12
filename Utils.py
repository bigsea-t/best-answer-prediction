import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
import re


def get_json(data_dir, n_ans):
    file_name = 'ans{}.dat'.format(n_ans)
    with open(data_dir + file_name) as file:
        data = json.load(file)
    return data


def remove_tags(s):
    return re.sub(r'\s*?<[^>]*?>\s*',' ',s)

def json_to_data_raw(js, max_score_th=3):
    texts = []
    Y = []
    for q in js:
        answers = q['Children']
        max_score = max([int(ans['Score']) for ans in answers])
        if max_score <= max_score_th:
            continue
        for ans in answers:
            t = ans['Body']
            y = int(ans['Score'])

            texts.append(t)
            Y.append(y)
    Y = np.array(Y)
    return texts, Y

def json_to_data(js, max_score_th=3):
    texts, Y = json_to_data_raw(js, max_score_th)
    texts = [remove_tags(i) for i in texts]
    vectorizer = CountVectorizer(stop_words='english', min_df=0.01, max_df=1.0)
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names()

    return X, Y, feature_names


def get_raw_data_score(data_dir, n_ans):
    return json_to_data_raw(get_json(data_dir, n_ans))

def get_data_score(data_dir, n_ans):
    return json_to_data(get_json(data_dir, n_ans))


def binarize_score(y, n_ans):
    n_samples, = y.shape
    n_questions = n_samples // n_ans

    if n_questions * n_ans != n_samples:
        raise ValueError('n_ans * n_questions != n_samples')

    y = y.reshape(n_questions, n_ans)
    binarized_y = np.zeros_like(y)
    binarized_y[np.arange(n_questions), y.argmax(axis=1)] = 1

    return binarized_y.flatten()


def reguralize_score(y, n_ans):
    n_samples, = y.shape
    n_questions = n_samples // n_ans

    if n_questions * n_ans != n_samples:
        raise ValueError('n_ans * n_questions != n_samples')

    y = y.reshape(n_questions, n_ans)
    y = y / y.max(axis=1)[:, np.newaxis]

    return y.flatten()


def split_data(X, Y, n_ans, train_size=0.75):
    n_samples, n_features = X.shape
    n_train = int(n_samples * train_size) // n_ans * n_ans
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def prec_at_1(Yprob, Y, n_ans):
    n_samples, = Yprob.shape
    n_questions = n_samples / n_ans

    Yprob = Yprob.reshape((n_questions, n_ans))
    Y = Y.reshape((n_questions, n_ans))

    Ypred = Yprob.argmax(axis=1)
    Y = Y.argmax(axis=1)

    prec = (Y == Ypred).sum() / n_questions

    return prec
