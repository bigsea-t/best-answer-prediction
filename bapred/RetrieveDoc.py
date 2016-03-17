import sys

from bapred.Utils import *

# This file is for searching occurance of specifiy words in dataset

def include_all_subs(string, subs):
    '''
    Parameters
    ----------
    strings: string

    subs: List(str)

    Returns: bool
    -------

    '''
    return sum([sub.lower() in string.lower() for sub in subs]) == len(subs)


def filter_with_substring(strings, subs):
    '''
    Parameters
    ----------
    strings: List(str)
        to be filtered

    subs: List(str)
        substring which has to be in filtered string

    Returns: List(Tuple(idx, str))
        filtered strings with original index
    -------

    '''

    out = [(i, s) for i, s in enumerate(strings) if include_all_subs(s, subs)]

    return out


def main(data_dir, n_ans, words):
    questions, answers, scores = get_raw_data_score(data_dir, n_ans, n_files=100)
    reg_scores = reguralize_score(scores, n_ans)

    filtered_ans = filter_with_substring(answers, words)

    with open('retrieved_{}.html'.format(str(words)), 'w') as f:
        f.write('<br/>include words:' + str(words))
        f.write('<br/>n_all_ans' + str(len(answers)))
        f.write('<br/>n_filtered_answers' + str(len(filtered_ans)))
        for i, ans in filtered_ans:
            # if reg_scores[i] != 1:
            #     continue
            q = questions[i // n_ans]
            f.write('<br/>score{}<br/>Question:<br/>{}<br/>Answer:<br/>{}<br/><hr><br/>'.format(reg_scores[i], q, ans))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Usage: python RetrieveDoc.py <data_dir>')
    data_dir = sys.argv[1]
    n_ans = 5
    # words = ['answers']
    words = ['edit']
    main(data_dir, n_ans, words)