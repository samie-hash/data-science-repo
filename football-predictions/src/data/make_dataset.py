# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

from sklearn.pipeline import Pipeline

from src.utils import utils, config
from src.features import build_features as buif

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--league', type=str, default='E0')
def main(input_filepath, output_filepath, league):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

        To run this file; navigate to this file location in the command line and run the command
        `python make_dataset.py ../../data/raw ../../data/processed/processed.csv`
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # read the data and merge
    data = utils.load_merge_data(input_filepath, keep=league, usecols=config.TRAIN_COLUMNS, seasons=6)

    # preprocess the data
    preprocessor = buif.Preprocessor()
    processed_data = preprocessor.fit_transform(data)
    
    # build features
    # pipeline = Pipeline([
    #     ('preprocessor', buif.Preprocessor()),

    # ])
    # save the processed dataset

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
