# visualize.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_club_avgss(data, club, ax):
    date = data.Date
    avgss = np.array([row.HAVGGS if row.HomeTeam == club else row.AAVGGS for idx, row in data.iterrows()])
    sns.lineplot(x=date[2:], y=avgss[2:], ax=ax, label=club)

def plot_avggs(data, club, season_label, title):
    season_data = data[data['SeasonLabel'] == season_label]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    if type(club) == str:
        mask = (season_data['HomeTeam'] == club) | (season_data['AwayTeam'] == club)
        plot_data = season_data[mask]
        plot_club_avgss(plot_data, club, ax)
        
    elif type(club) == list:
        for c in club:
            mask = (season_data['HomeTeam'] == c) | (season_data['AwayTeam'] == c)
            plot_data = season_data[mask]
            plot_club_avgss(plot_data, c, ax)
    plt.ylabel('Running Average Goals')
    plt.title(title, fontsize=16)
    plt.show()
    return fig

    