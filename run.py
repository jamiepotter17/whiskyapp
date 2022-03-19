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

from plotly.graph_objs import Bar, Scatter3d, Layout
from plotly.graph_objs.layout import Scene
from plotly.graph_objs.layout.scene import XAxis, YAxis, ZAxis
import joblib

from graphs import get_dataset_graphs, get_distance_graph
from all_whisk_clf import get_whisky_classifier

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
# n.b. whisky_classifier.pkl must be version created by running get_whisky_clf.py in whisky_project, not from within Jupyter Notebook. This is because the pickle file needs to not have any __main__ stuff in it.
df = pd.read_csv('branded.csv', index_col = 'Unnamed: 0')
whiskyclassifier = get_whisky_classifier()

distances_df = pd.read_csv('distances.csv', index_col = ['brand','distance_type'])

# index webpage displays visuals and receives user input text for model
@whiskyapp.route('/')
@whiskyapp.route('/index')
def index():

    # Gets graphs from graphs.py. Go there to edit.
    graphs = get_dataset_graphs(df)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

# render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@whiskyapp.route('/go')
def go():
    # save user input in query
    nosequery = request.args.get('nosequery', '')
    palatequery = request.args.get('palatequery', '')
    finishquery = request.args.get('finishquery', '')

    # Get the whiskyclassifier to produce its guess
    # Hacky code here unfortunately because I wrote the pipeline
    # so it would expect a 2-D array. Hence the empty row that
    # does absolutely nothing.
    test = np.array([['','',''],[nosequery, palatequery, finishquery]])
    guess = whiskyclassifier.predict(test)[1]

    # get distances_graph and encode to JSON object ready to send to template
    graph_data, graph_layout = get_distance_graph(distances_df, guess)
    data = { "data" : [graph_data], "layout": graph_layout}
    distance_graph = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    # This will render the go.html. Please see that file.
    return render_template(
        'go.html',
        nosequery=nosequery,
        palatequery=palatequery,
        finishquery=finishquery,
        guess=guess,
        distance_graph=distance_graph
    )

def main():
    whiskyapp.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
