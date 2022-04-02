# train_model.py
# This module holds utility classes and functions that trains a model
import time
import click
import numpy as np
import logging
import pandas as pd
import joblib as jbl

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib.cbook import boxplot_stats
from sklearn.base import BaseEstimator, TransformerMixin

from tqdm import tqdm # progress bar

class QuantileBasedAnomalyDetection(BaseEstimator, TransformerMixin):

    def __init__(self, k=1.5):
        self.k = k
        self.training_elapsed = 0.0

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self,deep=True):
        return {"k": self.k}

    def fit(self, X, y=None):
        # verify that data is 1-d
        start = time.time()
        if X.shape[1] > 1:
            raise ValueError('Data shape must be a 1-D array')

        stats = boxplot_stats(X.values, whis=self.k)
        self.lower_whisker, self.upper_whisker = stats[0].get('whislo'), stats[0].get('whishi')

        end = time.time()
        self.training_elapsed = end - start

        return self

    def predict(self, X, y=None):
        self.predictions = []

        for row in X.values:
            if self.__in_range(row):
                self.predictions.append(0) # normal
            else:
                self.predictions.append(1) # abnormal
        
        return self.predictions

    def __in_range(self, row):
        # verify that self.lower_whisker and self.upper_whisker exist
        if not (hasattr(self, 'lower_whisker') and hasattr(self, 'upper_whisker')):
            raise ValueError('You must fit the data to train data before calling predict')

        if row >= self.lower_whisker and row <= self.upper_whisker:
            return True
        return False

def evaluate_perf(true, pred, prettify=False):

    if prettify:
        return pd.DataFrame(columns=['f1_score', 'pre_score', 'rec_score', 'accuracy'], 
                            data=[[f1_score(true, pred), 
                                precision_score(true, pred), 
                                recall_score(true, pred), 
                                accuracy_score(true, pred)]]
        )

    return {'f1_score': f1_score(true, pred), 
            'pre_score': precision_score(true, pred), 
            'rec_score': recall_score(true, pred), 
            'accuracy': accuracy_score(true, pred)}

@click.command()
def build_model():
    
    logger = logging.getLogger(__name__)
    logger.info('creating model from processed data')

    # create model
    model = QuantileBasedAnomalyDetection()

    # loads the processed data from disk if available
    try:
        processed = pd.read_csv('../../data/processed/processed.csv')
    except FileNotFoundError:
        logger.error('Ensure you run "python make_dataset.py input_filepath output_filepath" before running this file')

    # fit the model to processed data
    col_length = processed.shape[1]

    processed = processed.sample(frac=1)
    X = processed.iloc[:, 0:col_length - 1]
    y = processed.iloc[:, col_length - 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)

    print('Training Performance')
    print(evaluate_perf(y_train, model.predict(X_train), prettify=True))
    print()
    print('Testing Performance')
    print(evaluate_perf(y_test, model.predict(X_test), prettify=True))

    # fine tune the model
    param_grid = {
        'k': np.linspace(1, 5, 50)
    }

    print('Starting Optimization')
    gs = GridSearchCV(model, param_grid=param_grid, scoring='f1', verbose=1)
    tqdm(gs.fit(X_train, y_train))

    best_model = gs.best_estimator_
    print('Optimization Done')

    print('Training Performance(Optimized Model)')
    print(evaluate_perf(y_train, best_model.predict(X_train), prettify=True))
    print()
    print('Testing Performance(Optimized Model)')
    print(evaluate_perf(y_test, best_model.predict(X_test), prettify=True))

    # save the model to disk
    print('.......Saving to disk')
    jbl.dump(best_model, '../../models/best_model.joblib')
    print('DONE!!!!')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    build_model()