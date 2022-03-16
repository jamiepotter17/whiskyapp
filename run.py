import json
import plotly
import pandas as pd
import numpy as np
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib

from graphs import get_graphs

whiskyapp = Flask(__name__)

# load in functions the models need
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

# load data and models
df = pd.read_csv('branded.csv', index_col = 'Unnamed: 0')
whiskyclassifier = joblib.load("whisky_classifier.pkl")
#maltclassifier = joblib.load("malt_classifier.pkl")



# index webpage displays visuals and receives user input text for model
@whiskyapp.route('/')
@whiskyapp.route('/index')
def index():

    # Gets graphs from graphs.py. Go there to edit.
    graphs = get_graphs(df)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

# render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@whiskyapp.route('/whiskygo')
def whiskygo():
    # save user input in query
    nosequery = request.args.get('nosequery', '')
    palatequery = request.args.get('palatequery', '')
    finishquery = request.args.get('finishquery', '')

    # Get the whiskyclassifier to produce its guess
    # Hacky code here unfortunately because I wrote the pipeline
    # so it would expect a two-D array. Hence the empty row that
    # does absolutely nothing.
    test = np.array([['','',''],[nosequery, palatequery, finishquery]])
    guess = whiskyclassifier.predict(test)[1]

    # This will render the gowhisky.html Please see that file.
    return render_template(
        'whiskygo.html',
        nosequery=nosequery,
        palatequery=palatequery,
        finishquery=finishquery,
        guess=guess
    )


#def main():
#    whiskyapp.run(host='0.0.0.0', port=3000, debug=True)


#if __name__ == '__main__':
#    main()
