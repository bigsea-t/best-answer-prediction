import re
import sys
from collections import Counter
from os import listdir

from bapred.utils import *

# This script is for organizing dataset by their number of answers.

def count_n_questions(posts):
    counter = Counter()
    for q in posts:
        counter[int(q['AnswerCount'])] += 1
    return counter


def organize_by_n_ans(raw):
    organized_posts = {}
    for q in raw:
        n_ans = q['AnswerCount']
        if n_ans not in organized_posts:
            organized_posts[n_ans] = []
        organized_posts[n_ans].append(q)

    return organized_posts


def dump_organized_posts(organized_posts, file_num):
    for num_ans, qs in organized_posts.items():
        out_file = 'ans{}-{}.dat'.format(num_ans, file_num)
        with open(data_dir + out_file, 'w') as file:
            json.dump(qs, file)


def organize_all(data_dir):
    files = [file for file in listdir(data_dir) if re.match(r'Posts[0-9]+\.dat', file) is not None]
    for file_num, file in enumerate(files):
        raw = get_post_from_file(file)
        organized_posts = organize_by_n_ans(raw)
        dump_organized_posts(organized_posts, file_num)


def get_post_from_file(file):
    out = []
    with open(data_dir + file) as fp:
        posts = json.load(fp)
    out += posts
    return out

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('only one argument is allowed (data_dir)')
    data_dir = sys.argv[1]

    organize_all(data_dir)

    print('done')