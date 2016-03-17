import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from os import listdir


def get_json(data_dir, n_ans, n_files=None):
    '''read json format dataset from `data_dir/ans`n_ans`.dat`'''
    if n_files is None:
        files = [file for file in listdir(data_dir) if re.match(r'ans{}\.dat'.format(n_ans), file) is not None]
        n_files = len(files)
    else:
        files = [file for file in listdir(data_dir) if re.match(r'ans{}(-[0-9]+)?\.dat'.format(n_ans), file) is not None]

    out = []
    for file_name in files[:n_files]:
        with open(data_dir + file_name) as file:
            data = json.load(file)
        out = out + data
    return out


def remove_tags(s):
    '''remove all HTML tags from `s`'''
    return re.sub(r'\s*?<[^>]*?>\s*',' ',s)


def json_to_data_raw(js, max_score_th=3):
    '''parse the json text and convert it into textual dataset 
    (and only consider the question with more than `max_score_th` votes).'''
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


def transform_raw_data(questions, answers, remove_tag=True, min_df=0.01, max_df=0.99):
    '''tokenize question and answer datasets'''
    n_questions = len(questions)
    n_answers = len(answers)
    n_each_answers = n_answers // n_questions
    assert(n_each_answers * n_questions == n_answers)

    if remove_tag:
        questions = [remove_tags(i) for i in questions]
        answers = [remove_tags(i) for i in answers]

    vectorizer = TfidfVectorizer(stop_words='english', min_df=min_df, max_df=max_df)

    vectorizer.fit(questions + answers)

    Xq = vectorizer.transform(questions)
    Xa = vectorizer.transform(answers)
    feature_names = vectorizer.get_feature_names()

    return Xq, Xa, feature_names


def get_raw_data_score(data_dir, n_ans, n_files=None):
    '''get the original answer-votes data'''
    return json_to_data_raw(get_json(data_dir, n_ans, n_files=n_files))


def binarize_score(y, n_ans):
    '''binarize the score of answers. Only the highest voted one of same question become 1, others 0.'''
    n_samples, = y.shape
    n_questions = n_samples // n_ans

    if n_questions * n_ans != n_samples:
        raise ValueError('n_ans * n_questions != n_samples')

    y = y.reshape(n_questions, n_ans)
    binarized_y = np.zeros_like(y)
    binarized_y[np.arange(n_questions), y.argmax(axis=1)] = 1

    return binarized_y.flatten()


def reguralize_score(y, n_ans):
    '''normalize the score of answers. Scale the highest-voted answer of same question to 1, others accordingly.'''
    n_samples, = y.shape
    n_questions = n_samples // n_ans

    if n_questions * n_ans != n_samples:
        raise ValueError('n_ans * n_questions != n_samples')

    y = y.reshape(n_questions, n_ans)
    y = y / y.max(axis=1)[:, np.newaxis]

    return y.flatten()


def split_data(Xq, Xa, Y, n_ans, train_size=0.75):
    '''split data to train and test data'''
    try:
        n_samples, n_features = Xq.shape
    except:
        n_samples = len(Xq)
    n_train_q = int(n_samples * train_size)
    n_train_a = n_train_q * n_ans

    return Xq[:n_train_q], Xa[:n_train_a], Y[:n_train_a],\
        Xq[n_train_q:], Xa[n_train_a:], Y[n_train_a:]

def prec_at_1(Yprob, Y, n_ans):
    '''calculate the accuracy of predicted result (`Yprob`)'''
    n_samples, = Yprob.shape
    n_questions = n_samples / n_ans

    Yprob = Yprob.reshape((n_questions, n_ans))
    Y = Y.reshape((n_questions, n_ans))

    Ypred = Yprob.argmax(axis=1)
    Y = Y.argmax(axis=1)

    prec = (Y == Ypred).sum() / n_questions

    return prec

def prec_at_1_model(model, X, y):
    '''calculate the accuracy of a model'''
    import bapred.ModelWrapper
    n_ans = 10
    Y = eval("bapred.ModelWrapper."+model.__class__.__name__+"Wrapper").predict_score(model, X)
    return prec_at_1(Y, y, n_ans)
