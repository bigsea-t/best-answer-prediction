import re

import numpy as np

from bapred.Utils import *

# Use random number as heuristic value
class DummyHeuristic:
    def __init__(self, n_ans):
        self.n_ans = n_ans

    def predict_score(self, X):
        return np.random.random(len(X))

# Use total number of characters as heuristic value
class CharLengthHeuristic:
    def __init__(self, n_ans):
        self.n_ans = n_ans
    
    def predict_score(self, X):
        return np.array([len(remove_tags(i)) for i in X])

# Use number of words as heurstic value
class WordLengthHeuristic:
    def __init__(self, n_ans):
        self.n_ans = n_ans
    
    def predict_score(self, X):
        return np.array([len(remove_tags(i).split(' ')) for i in X])

# Use number of characters as heuristic value and treat codes blocks' length as `cof`*length
class CharNCodeLengthHeuristic:
    def __init__(self, n_ans):
        self.n_ans = n_ans
        self.pattern = re.compile("<code>.*?</code>")
    
    def predict_score(self, X, cof=10):
        partitions = self.partition(X)
        return np.array([len(i)+cof*len(j) for i,j in partitions])
    
    def partition(self, texts):
        return [(remove_tags(self.pattern.sub("", i)), remove_tags("".join(self.pattern.findall(i)))) for i in texts]
              
# Use number of sentences as heuristic value
class SentenceLengthHeuristic:
    def __init__(self, n_ans):
        self.n_ans = n_ans
    
    def predict_score(self, X):
        return np.array([self.sentence_number(i) for i in X]) 
    
    def sentence_number(self, text):
        return len(list(filter(lambda i:i, re.split('[\.\?\!]\s+', remove_tags(text)))))

# Use number of sentences as heuristic value, but with code blocks' length as `cof`*length
class SentenceNCodeLengthHeuristic:
    def __init__(self, n_ans):
        self.n_ans = n_ans
        self.pattern = re.compile("<code>.*?</code>")
    
    def predict_score(self, X, cof=10):
        partitions = self.partition(X)
        return np.array([self.sentence_number(i)+cof*self.sentence_number(j) for i,j in partitions]) 
    
    def sentence_number(self, text):
        return len(list(filter(lambda i:i, re.split('[\.\?\!]\s+', remove_tags(text)))))
    
    def partition(self, texts):
        return [(remove_tags(self.pattern.sub("", i)), remove_tags("".join(self.pattern.findall(i)))) for i in texts]

# Use number of average length of sentences as heuristic value
class AvgSentenceLengthHeuristic:
    def __init__(self, n_ans):
        self.n_ans = n_ans
    
    def predict_score(self, X):
        return np.array([self.avg_sentence_length(i) for i in X]) 
    
    def avg_sentence_length(self, text):
        sentences = list(filter(lambda i:i, re.split('[\.\?\!]\s+', remove_tags(text))))
        return sum([len(i) for i in sentences])/len(sentences)