import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def get_json(data_dir, n_ans):
    file_name = 'ans{}.dat'.format(n_ans)
    with open(data_dir + file_name) as file:
        data = json.load(file)
    return data


def remove_tags(s):
    tag_re = re.compile(r'<[^>]+>|\n|\r')
    s = tag_re.sub(' ', s)
    return s


def json_to_data_inner(js, func_y, max_score_th=3):
    texts = []
    Y = []
    for q in js:
        answers = q['Children']
        max_score = max([int(ans['Score']) for ans in answers])
        if max_score <= max_score_th:
            continue
        for ans in answers:
            t = remove_tags(ans['Body'])
            score = int(ans['Score'])
            y = func_y(max_score, score)

            texts.append(t)
            Y.append(y)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=1, min_df=0)
    X = vectorizer.fit_transform(texts)
    Y = np.array(Y)
    return X, Y


def get_data_score(data_dir, n_ans):
    js = get_json(data_dir, n_ans)
    return json_to_data_inner(js, lambda mx, s: s)


def get_data_bin(data_dir, n_ans):
    js = get_json(data_dir, n_ans)
    return json_to_data_inner(js, lambda mx, s: 1 if mx==s else 0)


def get_data_prob(data_dir, n_ans):
    js = get_json(data_dir, n_ans)
    return json_to_data_inner(js, lambda mx, s: s / mx)


def split_data(X, Y, n_ans, train_size=0.75):
    n_samples, n_features = X.shape
    n_train = int(n_samples * train_size) // n_ans * n_ans
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def prec_at_1(Yprob, Y, n_ans):
    Yprob = Yprob[:,1]
    n_samples, = Yprob.shape
    n_questions = n_samples / n_ans

    Yprob = Yprob.reshape(n_questions, n_ans)
    Y = Y.reshape(n_questions, n_ans)

    Ypred = Yprob.argmax(axis=1)
    Y = Y.argmax(axis=1)

    prec = (Y == Ypred).sum() / n_questions

    return prec
