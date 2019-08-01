import sys
import pickle
import nltk

# Use the set_proxy statment ONLY if your organization proxy require authentication from non-brower applications 
# nltk.set_proxy('http://proxy.example.com:3128', ('USERNAME', 'PASSWORD'))
nltk.download('stopwords') # lists of stopwords
nltk.download(['punkt', 'wordnet']) # word tokeniser and lemmatisation
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    """
    Loads data from SQL-Lite database into a Pandas dataframe

    Args:
        database_filepath: database file path

    Returns:
        (DataFrame) X: features to be fed to ML model
        (DataFrame) Y: labels to be fed to ML model
        category_names list

    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_query('select * from cleanData', engine)

    X = df['message'].values
    y = df.drop(['id','message','original','genre'], axis=1)
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Tokenizes a given text message.

    Args:
        text: text string

    Returns:
        (str[]): array of clean tokens after normalization, lemmatization, and removing stopwords

    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Builds classification model
    Args:
        N/A

    Returns:
        return trained model
    
     """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=4)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model against a test dataset to get accuracy score.

    Args:
        model: Trained model
        X_test: Test features
        Y_test: Test labels
        category_names: String array of category names
    Returns:
        None
    """
    y_preds = model.predict(X_test)
    print(classification_report(y_preds, Y_test.values, target_names=category_names))
    print("**** Accuracy scores for each category *****\n")
    for i in range(36):
        print("Accuracy score for " + Y_test.columns[i], accuracy_score(Y_test.values[:,i],y_preds[:,i]))


def save_model(model, model_filepath):
    """
    Save the model as a Python "pickle" file

    Args:
        model: Trained model
        model_filepath: Path where the model will be saved 
    """
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
