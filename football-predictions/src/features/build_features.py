# build_features.py
# This module contains utility classes and functions that creates features and manipulate for a machine learning model

import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.base import TransformerMixin, BaseEstimator

@dataclass
class PositionStat:
    goals_for: int = 0
    goals_against: int = 0
    goals_diff: int = 0
    points: int = 0
    position: int = 0

class IntegerConvert(BaseEstimator, TransformerMixin):
    """
    This transformer converts the columns type specified to integer format
    """
    
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in self.columns:
            X[col] = pd.to_numeric(X[col])

        return X

class LeaguePosAdder(BaseEstimator, TransformerMixin):

    """
    This transformer manipulates the data and adds current league positions before a fixture is played
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        seasons = X.SeasonLabel.unique()
        X['HCLPOS'] = 0
        X['ACLPOS'] = 0

        for season in seasons:
            season_data = X[X['SeasonLabel'] == season]
            self.league_table = {}
        
        for club in season_data.HomeTeam.unique():
            self.league_table[club] = PositionStat()

        for idx, row in season_data.iterrows():
            # set the points
            X.loc[idx, 'HCLPOS'] = self.league_table[row.HomeTeam].position
            X.loc[idx, 'ACLPOS'] = self.league_table[row.AwayTeam].position
                
            # update team based on score result
            if row['FTR'] == 'H':
                self.league_table[row.HomeTeam] = self.__update_team(self.league_table[row.HomeTeam], 3, row['FTHG'], row['FTAG'])
                self.league_table[row.AwayTeam] = self.__update_team(self.league_table[row.AwayTeam], 0, row['FTAG'], row['FTHG'])
                
            if row['FTR'] == 'A':
                self.league_table[row.AwayTeam] = self.__update_team(self.league_table[row.AwayTeam], 3, row['FTAG'], row['FTHG'])
                self.league_table[row.HomeTeam] = self.__update_team(self.league_table[row.HomeTeam], 0, row['FTHG'], row['FTAG'])

            if row['FTR'] == 'D':
                self.league_table[row.HomeTeam] = self.__update_team(self.league_table[row.HomeTeam], 1, row['FTHG'], row['FTAG'])
                self.league_table[row.AwayTeam] = self.__update_team(self.league_table[row.AwayTeam], 1, row['FTAG'], row['FTHG'])

            # sort the self.league_table
            self.league_table = dict(sorted(self.league_table.items(), key=lambda x: (x[1].points, -x[1].goals_diff), reverse=True))

            # set the position
            pos = 1
            for key, value in self.league_table.items():
                self.league_table[key].position = pos
                pos += 1

        return X

    def transform(self, X, y=None):
        return self.league_table

    def __update_team(self, team, points, goals_for, goals_against):
        team.points += points
        team.goals_for += goals_for
        team.goals_against += goals_against
        team.goals_diff = team.goals_for - team.goals_against
        
        return team

    
    
    