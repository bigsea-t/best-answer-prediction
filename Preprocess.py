from Utils import *
import sys
from os import listdir
from collections import Counter
import re


def count_n_questions(posts):
    counter = Counter()
    for q in posts:
        counter[int(q['AnswerCount'])] += 1
    return counter


def organize_by_n_ans(data_dir, posts):
    organized_posts = {}
    for q in posts:
        n_ans = q['AnswerCount']
        if n_ans not in organized_posts:
            organized_posts[n_ans] = []
        organized_posts[n_ans].append(q)

    for num_ans, qs in organized_posts.items():
        out_file = 'ans{}.dat'.format(num_ans)
        with open(data_dir + out_file, 'w') as file:
            json.dump(qs, file)


def get_posts(data_dir):
    out = []
    files = [file for file in listdir(data_dir) if re.match(r'Posts[0-9]+\.dat', file) is not None]
    for file in files:
        with open(data_dir + file) as fp:
            posts = json.load(fp)
        out += posts
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('only one argument is allowed (data_dir)')
    data_dir = sys.argv[1]

    posts = get_posts(data_dir)
    organize_by_n_ans(data_dir, posts)

    print('coutns by num answers')
    print(count_n_questions(posts))
