# Whisky Project

## Summary

This is a natural language, machine learning project that classifies whiskies based on tasting notes provided by the user. It was trained on data harvested from Reddit's whisky community through Reddit's API.

This repository contains the code needed for the web application hosted on Heroku. The repository containing the bulk of the background work and code for Whisky Project is to be found at [whisky_project](https://github.com/jamiepotter17/whisky_project).

## Instructions:

The completed app is hosted on Heroku at [https://whiskyproject.herokuapp.com](https://whiskyproject.herokuapp.com). Please go there to view the app.

To use, simply enter your tasting notes for the nose, palate, and finish of your whisky, and click 'Guess Whisky Brand'. For example, my review of a typical Laphroaig (my favourite whisky!) would be:

* Nose - medicinal, peat, smoky
* Palate - sweet, mint, liquorice, peat
* Finish - sweet, salty, long

## Background

This is the Capstone Project for Udacity's [Become a Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025) Nanodegree Program. It was commenced on 2012-02-28. Here the brief was fairly open, but the essential idea was to follow the Data Science Process from start to finish on a project of my own design. That is, to:

1. **Define** the problem you want to solve and investigate potential solutions.
2. **Analyse** the problem through visualisations and data exploration.
3. **Implement** algorithms and metrics, documenting any preprocessing, refinement, and post-processing steps along the way.
4. **Collect results**, and draw conclusions about whether your implementation adequately addresses the problem.

The final project could be in the form of either a blog post documenting all of the steps from start to finish of your project, or a deployment of a web application (or something that can be run on a local machine).

In my case, I decided that I was most interested in developing a web application that would allow users to enter tasting notes on a whisky, and it would predict what whisky it was. Other potential functionality I thought I might include was generating word clouds for particular brands of whisky and suggest similar whiskies.

## Data

The data I have gathered was taken from [Reddit's API](https://www.reddit.com/dev/api) using [PRAW](https://praw.readthedocs.io/en/stable/), a Python package that provides a useful wrapper when working with the API. A link to a list of reviews called the '[Whisky Review Archive](https://docs.google.com/spreadsheets/d/1X1HTxkI6SqsdpNSkSSivMzpxNT-oeTbjFFDdEkXD30o/edit#gid=695409533&fvid=484110565)' [Google Docs link] is made available on the [/r/whisky](https://www.reddit.com/r/whisky/) subreddit. I downloaded this spreadsheet file as a csv, and then augmented it as best I could so that the each row has the original review text included under the 'review' column.

## Ethics

In order to address any ethical concerns about the use of data, I note the following:

* All data were gathered via use of Reddit's API, using PRAW's standard downloading rate. No scraping was used.
* Reddit usernames are included in the spreadsheet, and Reddit users understand that their usernames are public-facing avatars. Thus, it is not disclosing private information to include this information. Nonetheless, seeing as it is not an important part of my project, I have removed username data from my dataset once I move beyond the data gathering stage.
* I will make users of /r/whisky aware of the existence of the app in the hopes that they will gain value from it.
* The app is for personal use and learning purposes only. It is not intended for commercial purposes. People are licenced to use it as they see fit (see licence.md)

## File list:

* README.MD - this file.
* branded.csv - csv file ready to be fed into ML algorithm.
* distances.csv  - csv file with Euclidean distances between the whiskies. Use index_col = ['brand','distance_type'] to open as it has a multi-index.
* ./static/images/whiskies.jpg - picture of Brora whiskies I took whilst visiting the fine distillery at Clynelish. Sadly, I've still not tasted any Brora myself.
* ./templates/master.html - the landing page of the app where you enter your notes. Also displays the graphs showing summary info about training set.
* ./templates/go.html - will display the result of the model prediction and then a 3d scatterplot showing other whiskies with (nose distance, palate distance, finish distance) coordinates so you can identify similar whiskies.
* graphs.py - module that contains the visualisations to be loaded in.
* all_whisk_clf.py - module that contains function for loading in the whisky classifier model.
* nltk.txt - instruction to heroku to download stopwords from nltk.
* Procfile - instruction to heroku so it knows what to run.
* requirements.txt - contains only the requirements for a minimal virtual environment needed to run the app.
* run.py - the main file that gets run. Use 'python run.py' to host locally.
