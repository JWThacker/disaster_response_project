import json
import plotly
import pandas as pd
import plotly.express as px

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    ''' Tokenize textual data
    
        params:
            text - a string
        returns:
             clean_tokens - clean tokens ready for TFIDF vectorizer
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/message_categories.db')
df = pd.read_sql_table('message_categories', engine)

# load model
model = joblib.load("../models/model.sav")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    response_class_df = df.loc[:, 'related':'direct_report'].sum().reset_index(name='frequency')
    response_class_df.rename(columns={'index':'class'}, inplace=True)
    # create visuals
    agg_df = pd.DataFrame(data={'counts': genre_counts, 'names':genre_names})
    fig1 = px.bar(data_frame=agg_df, x='names', y='counts', title='Overview of Training Dataset')
    fig2 = px.bar(data_frame=response_class_df, x='class', y='frequency', title='Response Class Bar Graph')
    graphs = [fig1, fig2]
    
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
