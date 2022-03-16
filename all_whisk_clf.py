import pandas as pd
import numpy as np
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import joblib

def tokenise_and_stem_text(text):
    '''
    INPUTS:
    text (string) - what you want to be lemmatised
    OUTPUTS:
    lemmas (list) - list of lemmatised next
    '''
    # Import stopword list and update with a few of my own
    stopword_list = stopwords.words("english")
    [stopword_list.append(i) for i in ['nose', 'palate', 'taste', 'finish']]

    # Normalise text - remove numbers too as we don't need them
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())

    # tokenise
    words = text.split()

    # Checks it's a word and removes stop words
    words = [word for word in words if word not in stopword_list]

    # Create stemmer object
    stemmer = PorterStemmer()

    # Add lemmas
    lemmas = []
    for word in words:
        lemmas.append(stemmer.stem(word))

    return lemmas

def getnose(array):
    return array[:,0]

def getpalate(array):
    return array[:,1]

def getfinish(array):
    return array[:,2]

def get_whisky_classifier():
    whiskyclassifier = joblib.load("whisky_classifier.pkl")
    return whiskyclassifier
