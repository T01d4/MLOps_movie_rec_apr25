# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Define input file paths
    input_filepath_scores = os.path.join(input_filepath, "genome-scores.csv")
    input_filepath_gtags = os.path.join(input_filepath, "genome-tags.csv")
    input_filepath_movies = os.path.join(input_filepath, "movies.csv")
    input_filepath_ratings = os.path.join(input_filepath, "ratings.csv")
    input_filepath_tags = os.path.join(input_filepath, "tags.csv")
    
    # Call the main data processing function with the provided file paths
    process_data(input_filepath_scores, input_filepath_gtags, input_filepath_movies, input_filepath_ratings, input_filepath_tags, output_filepath, logger)


def process_data(input_filepath_scores, input_filepath_gtags, input_filepath_movies, input_filepath_ratings, input_filepath_tags, output_filepath, logger):
    try:
        # Importing datasets with optimized memory usage
        logger.info("Loading datasets...")
        df_scores = pd.read_csv(input_filepath_scores, sep=",", dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"})
        df_gtags = pd.read_csv(input_filepath_gtags, sep=",", dtype={"tagId": "int32", "tag": "string"})
        df_movies = pd.read_csv(input_filepath_movies, sep=",", usecols=["movieId", "title", "genres"], dtype={"movieId": "int32", "title": "string", "genres": "string"})
        df_ratings = pd.read_csv(input_filepath_ratings, sep=",", usecols=["userId", "movieId", "rating"], dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})
        df_tags = pd.read_csv(input_filepath_tags, sep=",", usecols=["userId", "movieId", "tag"], dtype={"userId": "int32", "movieId": "int32", "tag": "string"})
        
        # Merging datasets incrementally
        logger.info("Merging datasets...")
        df = pd.merge(df_scores, df_gtags, on='tagId', how='left')
        df = pd.merge(df, df_movies, on='movieId', how='left')
      #  df = pd.merge(df, df_ratings, on='movieId', how='left')
      #  df = pd.merge(df, df_tags, on='movieId', how='left')

        # Drop rows with missing values in specific columns
        col_to_drop_lines = [ "relevance"]  # Define columns to check for NaN
        df = df.dropna(subset=col_to_drop_lines, axis=0)

        # Split features and target
        target = df['movieId']
        feats = df.drop(['movieId', 'tagId'], axis=1)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)

        # Create output folder if necessary
        if not os.path.exists(output_filepath):
            os.makedirs(output_filepath)

        # Save the train-test split data
        for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
            output_file = os.path.join(output_filepath, f'{filename}.csv')
            file.to_csv(output_file, index=False)

        logger.info("Data processing completed successfully.")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing file: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()