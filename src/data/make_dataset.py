# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

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
        
        # Merging datasets
        logger.info("Merging datasets...")
        df = pd.merge(df_ratings, df_scores , on='movieId', how='left')


        # Drop rows with missing values in specific columns
        col_to_drop_lines = ["rating"]  # Ensure 'rating' is used for the matrix
        logger.info(f"Dropping rows with missing values in columns: {col_to_drop_lines}")
        df = df.dropna(subset=col_to_drop_lines, axis=0)

        # Create a pivot table to generate the movie matrix
        logger.info("Creating movie matrix...")
        movie_matrix = df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

        # Save the movie matrix to a CSV file
        output_file = os.path.join(output_filepath, 'movie_matrix.csv')
        movie_matrix.to_csv(output_file)

        logger.info(f"Movie matrix saved to {output_file}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing file: {e}")
    except KeyError as e:
        logger.error(f"Key error: {e}")
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