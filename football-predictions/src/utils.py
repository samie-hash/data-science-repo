# utils.py
# This are functions that are not necessarily in any subcategory

import os
import string
import pandas as pd
import numpy as np
import pandas as pd
from datetime import timedelta

TRAIN_COLUMNS = ['Div','Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','Referee','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']
ELO_COLUMNS =  ['Div','Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR']

def load_merge_data(path:str, keep, usecols:list=None, seasons='all') -> pd.DataFrame:
    """
    Merges the football data in a specific folder into a single dataset

    Parameters
    ----------
    path: str
        The path to read the datasets from
    usecols: list
        The columns to read from the dataframe defaults to None
    seasons: list or int 'all' or int
        The seasons to read from the path, defaults to 'all' and returns the entire seasons
        from 2017 or when a integer value is pass, it returns the last n seasons
    keep: league to return i.e E0 for Premier league, E1 for Championship e.t.c

    Returns
    -------
    data_frame_model: pd.Datframe
        The merged dataframe
    """

    data_frame_model = pd.DataFrame(columns=usecols)

    if seasons == 'all':
        start = 0
    elif int(seasons) <= len(os.listdir(path)):
        start= len(os.listdir(path)) - (int(seasons) - 1)
    else:
        raise ValueError('"seasons" does not match the number of seasons available. Either pass in "all" to get the entire data or pass a numeric number to get the last "n" seasons')
    for dir in os.listdir(path)[start:]: # List all directories in the base dir
        if not os.path.isdir(f'{path}/{dir}'):
            continue
        for d in os.listdir(f'{path}/{dir}'): 
            file_path = f'{path}/{dir}/{d}'
            if file_path.split('/')[-1].split('.')[0] == keep:
                file = pd.read_csv(file_path, usecols=usecols)
                file['SeasonLabel'] = dir
                data_frame_model = pd.concat([data_frame_model, file], ignore_index=True)
    
    return data_frame_model

def remove_puntuations(x):
    return x.translate(str.maketrans('', '', string.punctuation))

def date_to_regular_fmt(date_str: str):
    date_split = date_str.split('/')
    month = date_split[1]
    day = date_split[0]
    year = date_split[2]

    if len(year) == 2 and year[0] == '9':
        year = f'19{year}'
    elif len(year) == 2 and year[0] != '9':
        year = f'20{year}'

    return f'{month}/{day}/{year}'

def calc_p(elort1, elort2, gf, ga, k=20):
    gd = np.abs(gf - ga)
    g = 0; w = 0.5; we = 0.5; minus_dr = 0
    
    if (gd == 0) or (gd == 1):
        g = 1
    elif gd == 2:
        g = 3 / 2
    else:
        g = (11 + gd) / 8
    
    if gf > ga:
        w = 1
    elif ga > gf:
        w = 0

    minus_dr = -((elort1) - elort2)

    we = 1 / ((10 ** (minus_dr/600)) + 1)

    p = int(k * (g * (w - we)))
    
    return p

def build_elo(league):
    # read the data
    data = load_merge_data('../data/raw/', keep=league, usecols=ELO_COLUMNS, seasons='all')

    # drop missing data
    data.dropna(axis=0, inplace=True)

    data['Date'] = data['Date'].apply(lambda x: date_to_regular_fmt(x))
    data['Date'] = pd.to_datetime(data['Date'])

    data['HELORT'] = 1400
    data['AELORT'] = 1400
    
    data.HomeTeam = data.HomeTeam.apply(lambda x: remove_puntuations(x))
    data.AwayTeam = data.AwayTeam.apply(lambda x: remove_puntuations(x))

    unique_teams = data.HomeTeam.unique()
    elort_dict = {}

    for team in unique_teams:
        elort_dict[team] = 1400


    for idx, row in data.iterrows():
        # current home team elort
        ht_current_elort = elort_dict[row.HomeTeam]
        at_current_elort = elort_dict[row.AwayTeam]
        data.loc[idx, 'HELORT'] = ht_current_elort
        data.loc[idx, 'AELORT'] = at_current_elort
        
        # update the current_elort based on the result for home_team
        fthg = row.FTHG
        ftag = row.FTAG

        home_p = calc_p(ht_current_elort, at_current_elort, fthg, ftag, k=20)
        away_p = calc_p(at_current_elort, ht_current_elort, ftag, fthg, k=20)

        ht_new_elort = ht_current_elort + home_p
        at_new_elort = at_current_elort + away_p
        
        elort_dict[row.HomeTeam] = ht_new_elort
        elort_dict[row.AwayTeam] = at_new_elort

    return data

def integer_convert(X, columns):
    for col in columns:
            X[col] = pd.to_numeric(X[col])

    return X

def rows_extractor(X, shift=6):
    """Extracts season data after shifting the date by 'shift' weeks """

    seasons = X.SeasonLabel.unique()
    data = pd.DataFrame()

    for season in seasons:
        # grab season data
        mask = X['SeasonLabel'] == season
        temp = X[mask]

        min_date = temp.Date.min()
        shift_date = min_date + timedelta(weeks=shift)

        mask = temp['Date'] >= shift_date
        data = pd.concat([data, temp[mask]], ignore_index=True)

    return data
if __name__ == '__main__':
    data = build_elo('E0')
    print(data.head())