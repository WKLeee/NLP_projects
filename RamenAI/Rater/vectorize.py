import json
import config as conf
import re

import nlp
import numpy as np
import data_processing as dp
import sklearn
from sklearn import decomposition

class WordDict(object):
    def __init__(self, sc):
        self.wordmap = {}
        self.counter = 0
        self.wordstatistic = []
        self.sentences_count = sc

    def add_word(self, word, sentence_index):
        if word in self.wordmap:
            cur_counter = self.wordmap[word]
            info = self.wordstatistic[cur_counter]
            info["occur"].append(sentence_index)
        else:
            self.wordmap[word] = self.counter
            self.wordstatistic.append({})
            self.wordstatistic[self.counter] = {"word": word, "occur": [sentence_index]} 
            self.counter += 1

    def vectorize(self):
        m = len(self.wordmap)
        n = self.sentences_count
        vec = np.zeros((n, m))
        for j in range(self.counter):
            for i in self.wordstatistic[j]["occur"]:
                vec[i][j] += 1
        return vec

def is_pure_English(word):
    res = re.match('[A-Za-z]+$', word)
    if res:
        return True
    else:
        return False


def stemming(sentences):
    n = len(sentences)
    lt = nlp.NLP() # language tool
    wd = WordDict(n) # dictionary

    for i in range(n):
        words = lt.tokenize(sentences[i])
        tags = lt.pos_tag(words)
        for wt in tags:
            if conf.TAG_MODE == 0 or wt[1] in conf.GOOD_TAGS:
                w = wt[0]
                # lemmatize word
                if w.lower() in lt.stopwords:
                    continue
                lmtz_w = lt.lemmatize(w)
                if lmtz_w == None:
                    continue
                if not is_pure_English(lmtz_w):
                    continue
                if lmtz_w.lower() in lt.stopwords:
                    continue
                lmtz_w = lt.stem(lmtz_w).lower()
                wd.add_word(lmtz_w, i)
    # vectorize
    vec = wd.vectorize()
    return vec, wd

def stemming_new_sentences(new_sentences, wd):
    lt = nlp.NLP()
    n = len(new_sentences)
    n_words = len(wd.wordmap)
    vec = np.zeros((n, n_words))
    for i in range(n):
        words = lt.tokenize(new_sentences[i])
        tags = lt.pos_tag(words)
        for wt in tags:
            if conf.TAG_MODE == 0 or wt[1] in conf.GOOD_TAGS:
                w = wt[0]
                # lemmatize word
                if w.lower() in lt.stopwords:
                    continue
                lmtz_w = lt.lemmatize(w)
                if lmtz_w == None:
                    continue
                if not is_pure_English(lmtz_w):
                    continue
                if lmtz_w.lower() in lt.stopwords:
                    continue
                lmtz_w = lt.stem(lmtz_w).lower()
                if lmtz_w in wd.wordmap:
                    word_index = wd.wordmap[lmtz_w]
                    vec[i][word_index] += 1

    return vec


def get_vectors(n_dimension=0):
    if conf.TEST == 0:
        review_path = conf.REVIEW_DATA_PATH
    else:
        review_path = conf.MINI_DATA_PATH
    data = dp.load_file(review_path)
    n = len(data)
    sentences = []
    labels = np.zeros((n, 1))
    for i in range(n):
        sentences.append(data[i]["text"])
        labels[i] = data[i]["stars"]
    # stem the sentences
    vec, wordDict = stemming(sentences)
    # PCA
    if n_dimension == "mle" or n_dimension > 0:
        # use pca to reduce the dimension
        pca_model = sklearn.decomposition.PCA(n_components=n_dimension, copy=True, whiten=False)
        newX = pca_model.fit_transform(vec)
        return newX, labels, wordDict, pca_model
    return vec, labels, wordDict, None


def get_vectors_new_sentences(pca_model, new_sentences, wd):
    vec = stemming_new_sentences(new_sentences, wd)
    if pca_model != None:
        newX = pca_model.transform(vec)
        return newX
    return vec


if __name__ == "__main__":
    import time
    start_time = time.time()
    vec, labels, word_dict, pca_model = get_vectors()
    print(vec.shape)
    new_sentences = ["good great burritos!"]
    a = get_vectors_new_sentences(pca_model, new_sentences, word_dict)
    print(np.sum(a))
    end_time = time.time()
    print(end_time-start_time)

