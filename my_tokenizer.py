# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 22:40:53 2017

@author: Administrator
"""

from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer

from nltk.tokenize.api import StringTokenizer


class MyTokenizer(object):

    def __init__(self, language='english', stemmer='porter'):
        switcher = {
            'porter' : PorterStemmer(),
            'lancaster' : LancasterStemmer(),
            'snowball' : SnowballStemmer(language)
        }
        self.stemmer = switcher.get(stemmer)

    def __call__(self, s):
        text = unicode.split(s)
        for i in range(len(text)):
            text[i] = self.stemmer.stem(text[i])
        return text
        
a = MyTokenizer()