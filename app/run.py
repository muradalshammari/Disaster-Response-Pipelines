import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from contextlib import redirect_stdout
import nltk
import os

app = Flask(__name__)


class NumUpperExtractor(BaseEstimator, TransformerMixin):
    # extractor/transformer to find all uppercase words
    def transform(self, X, y=None):
        X_transformed = pd.Series(X).apply(
            lambda x: len([x for x in x.split() if x.isupper()]))
        return pd.DataFrame(X_transformed)

    def fit(self, X, y=None):
        return self


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def tokenize(text):
    '''
    INPUT:
        TEXT (string) : text to tokenize/lemmatize
    OUTPUT:
        CLEAN_WORDS (list) : list of tokenized/cleaned words
    '''

    detected_urls = re.findall(url_regex, text)  # find urls
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')  # replace urls

    tokens = word_tokenize(
        text)  # tokenizer object, not capitalised as it is a class method

    words = [word for word in tokens if word not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()  # parent class lemmatizer object

    clean_words = []  # empty list for results

    for word in words:
        clean_word = lemmatizer.lemmatize(
            word).lower().strip()  # return lemmatized words

        clean_words.append(clean_word)  # append cleaned/lemmatized string

    return clean_words


with redirect_stdout(open(os.devnull, "w")):
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    ### data for visualizing category counts.
    label_sums = df.iloc[:, 4:].sum()
    label_names = list(label_sums.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=label_names,
                    y=label_sums,
                )
            ],

            'layout': {
                'title': 'Distribution of labels/categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {

                },
            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]

    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=8000, debug=True)


if __name__ == '__main__':
    main()
