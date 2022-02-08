import sys
import re
import pickle
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM message_categories', engine)
    X = df.loc[:, 'message']
    Y = df.loc[:, 'related':'direct_report']
    return X, Y, Y.columns


def tokenize(text):
    
    '''Tokenize data to be used in TFIDF vectorizers
    
       params:
           text - a string to be tokenized
       returns:
           clean_tokens -  tokens from text ready for TFIDF vectorizers
    '''
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    clean_tokens = list()
    tokens = word_tokenize(text)
    
    removed_stopwords = [token for token in tokens if token not in stopwords.words('english')]
    
    clean_tokens = [lemmatizer.lemmatize(word, pos='v').lower().strip() for word in removed_stopwords]
    clean_tokens = [lemmatizer.lemmatize(token, pos='n').lower().strip() for token in clean_tokens]
    return clean_tokens

def build_model():
    
    '''Build the machine learning model using grid search
    
       Only one parameter is varied to reduce the time needed for training.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf_trans', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=5)))
    ])
    
    parameters = {
    'clf__estimator__min_samples_split':[2, 5, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names=None):
    
    '''Evaluate the model's performance
    '''
    Y_pred = model.predict(X_test)
    for i, class_ in enumerate(Y_test.columns):
        print(class_, classification_report(Y_test.loc[:, class_].values, Y_pred[:, i]))


def save_model(model, model_filepath):
    '''Save the model as a .pkl file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()