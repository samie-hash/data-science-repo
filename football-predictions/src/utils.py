# utils.py
# This are functions that are not necessarily in any subcategory

import os
import pandas as pd

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

