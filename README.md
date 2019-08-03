
# Disaster-Response-Pipeline
 Disaster Response Pipeline Project is part of Udacity Data-Scientist NanoDegree program

## Table of Contents
1. [Overview](#Overview)
2. [Project Components](#files)
3. [Installation](#Installation)
4. [Example](#Example)



### Overview <a name="Overview"></a>

In the Project Workspace, you'll analyze a data set containing real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.


### Project Components <a name="files"></a>

1. ETL Pipeline

The Python script, process_data.py, contains a data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline

The Python script, train_classifier.py, has a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App

- It displays results in a Flask web app


### Installation <a name="Installation"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Example <a name="Example"></a>

> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

> python train_classifier.py ../data/DisasterResponse.db classifier.pkl

> python run.py