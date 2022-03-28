# train_model.py
# This module holds utility classes and functions that trains a model
import numpy as np
import pandas as pd
import joblib as jbl

import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from matplotlib.cbook import boxplot_stats
from sklearn.base import BaseEstimator, TransformerMixin

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
        if len(X.shape) == 2:
            raise ValueError('Data shape must be a 1-D array')

        stats = boxplot_stats(X.values, whis=self.k)
        self.lower_whisker, self.upper_whisker = stats[0].get('whislo'), stats[0].get('whishi')

        end = time.time()
        self.training_elapsed = end - start

        return self

    def predict(self, X, y=None):
        self.predictions = []

        for row in X:
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



