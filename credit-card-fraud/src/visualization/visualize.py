# visualize.py
# This module holds utility classes and functions for visualization
import pandas as pd
import matplotlib.pyplot as plt

def quantile_model_viz(X, k=1.5, output=None, ylabel=''):
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    if len(X.shape) == 2:
        raise ValueError('Data shape must be a 1-D array')

    plt.boxplot(X.values, whis=k)
    plt.xlabel('quantile_model')
    plt.title(f'Box plot of {X.name} with whisker value: {k}', fontsize=14)
    plt.ylabel(ylabel)
    plt.show()

    if output:
        fig.savefig(fname=output, dpi=80)
