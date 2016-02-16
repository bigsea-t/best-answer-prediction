import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from os import listdir


def get_json(data_dir, n_ans):
    files = [file for file in listdir(data_dir) if re.match(r'ans{}-[0-9]\.dat'.format(n_ans), file) is not None]
    out = []
    for file_name in files:
        with open(data_dir + file_name) as file:
            data = json.load(file)
        out = out + data
    return out


def remove_tags(s):
    return re.sub(r'\s*?<[^>]*?>\s*',' ',s)


def json_to_data_raw(js, max_score_th=3):
    questions = []
    answers = []
    Y = []
    for q in js:
        ans_json = q['Children']

        max_score = max([int(ans['Score']) for ans in ans_json])
        if max_score <= max_score_th:
            continue

        questions.append(q['Title'] + ' ' + q['Body'])

        for ans in ans_json:
            t = ans['Body']
            y = int(ans['Score'])

            answers.append(t)
            Y.append(y)

    scores = np.array(Y)

    return questions, answers, scores


def transform_raw_data(questions, answers):
    n_questions = len(questions)
    n_answers = len(answers)
    n_each_answers = n_answers // n_questions
    assert(n_each_answers * n_questions == n_answers)

    questions = [remove_tags(i) for i in questions]
    answers = [remove_tags(i) for i in answers]

    # vectorizer = CountVectorizer(stop_words='english', min_df=0.01, max_df=1.0)
    vectorizer = TfidfVectorizer(stop_words='english', min_df=0.01, max_df=0.99)

    vectorizer.fit(questions + answers)

    Xq = vectorizer.transform(questions)
    Xa = vectorizer.transform(answers)
    feature_names = vectorizer.get_feature_names()

    return Xq, Xa, feature_names


def get_raw_data_score(data_dir, n_ans):
    return json_to_data_raw(get_json(data_dir, n_ans))


# don't use this fucntion, just separete them

# def get_data_score(data_dir, n_ans):
#     return json_to_data(get_json(data_dir, n_ans))


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


def split_data(Xq, Xa, Y, n_ans, train_size=0.75):
    n_samples, n_features = Xq.shape
    n_train_q = int(n_samples * train_size)
    n_train_a = n_train_q * n_ans

    return Xq[:n_train_q], Xa[:n_train_a], Y[:n_train_a],\
        Xq[n_train_q:], Xa[n_train_a:], Y[n_train_a:]



def prec_at_1(Yprob, Y, n_ans):
    n_samples, = Yprob.shape
    n_questions = n_samples / n_ans

    Yprob = Yprob.reshape((n_questions, n_ans))
    Y = Y.reshape((n_questions, n_ans))

    Ypred = Yprob.argmax(axis=1)
    Y = Y.argmax(axis=1)

    prec = (Y == Ypred).sum() / n_questions

    return prec
