#coding=utf-8

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords

class NLP(object):
    def __init__(self):
        #self.stemmer = PorterStemmer()
        self.stemmer = SnowballStemmer("english")
        self.lemmatizer = WordNetLemmatizer()
        self.words = {}
        self.stopwords = stopwords.words('english')
        # f = open('stopwords_used.txt', 'r')
        # self.stopwords = []
        # content = f.read()
        # for line in content.split('\n'):
        #     if len(line) > 0:
        #         self.stopwords.append(line.strip())
        # f.close()

    def tokenize(self, sentence):
        '''
        rtype: list
        '''
        return word_tokenize(sentence)

    def stem(self, word):
        '''
        rtype: str
        '''
        return self.stemmer.stem(word)

    def bad(self, word):
        if word in self.words:
            return self.words[word]
        for c in word:
            if ord(c) >= 128:
                self.words[word] = True
                return True
        self.words[word] = False
        return False

    def lemmatize(self, word):
        '''
        rtype: str
        '''
        if self.bad(word):
            return None
        return self.lemmatizer.lemmatize(word)

    def pos_tag(self, words):
        '''
        rtype: tuple
        '''
        return nltk.pos_tag(words)

    def is_English_word(self, word):
        if not wordnet.synsets(word):
            return False
        return True

    def get_synonyms(self, word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                ln = lemma.name()
                if not ln in synonyms:
                    synonyms.append(lemma.name())
        return synonyms

    def get_antonyms(self, word):
        antonyms = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                if l.antonyms():
                    ant = l.antonyms()[0].name()
                    if not ant in antonyms:
                        antonyms.append(ant)
        return antonyms

if __name__ == "__main__":
    s = NLP()
    print(s.stopwords)
    print(type(s.stopwords))