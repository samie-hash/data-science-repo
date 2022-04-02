# -*- coding: utf-8 -*-
import sys
sys.path.append('..')

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import features.build_features as buif

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

        To run this file; navigate to this file location in the command line and run the command
        `python make_dataset.py ../../data/raw/creditcard.csv ../../data/processed/processed.csv`
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # read the input file
    data = pd.read_csv(input_filepath)

    # create the data processing pipeline
    columns = [buif.correlation_columns(data, 'Class', k=.2)[0]]
    columns.extend(['Class'])

    pipeline = Pipeline(steps=[
        ('column_extractor', buif.ColumnExtractor(columns)),
    ])

    # fit the pipeline to data
    processed = pipeline.fit_transform(data)

    # save the processed data to disk
    processed.to_csv(output_filepath, index=None)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
