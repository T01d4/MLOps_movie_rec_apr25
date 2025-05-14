import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
from tqdm import tqdm


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
    process_data(input_filepath_scores, input_filepath_gtags,
                 input_filepath_movies, input_filepath_ratings,
                 input_filepath_tags, output_filepath, logger
                 )


def process_data(input_filepath_scores, input_filepath_gtags,
                 input_filepath_movies, input_filepath_ratings,
                 input_filepath_tags, output_filepath, logger
                 ):
    try:
        # Importing datasets with optimized memory usage
        logger.info("Loading datasets...")
        df_scores = pd.read_csv(input_filepath_scores, sep=",",
                                dtype={
                                    "movieId": "int32",
                                    "tagId": "int32",
                                    "relevance": "float32"
                                    })
        # unused
        df_gtags = pd.read_csv(input_filepath_gtags, sep=",",
                               dtype={
                                   "tagId": "int32",
                                   "tag": "string"
                                   })
        # unused
        df_movies = pd.read_csv(input_filepath_movies, sep=",",
                                usecols=["movieId", "title", "genres"],
                                dtype={
                                    "movieId": "int32",
                                    "title": "string",
                                    "genres": "string"
                                    })
        df_ratings = pd.read_csv(input_filepath_ratings, sep=",",
                                 usecols=["userId", "movieId", "rating"],
                                 dtype={
                                     "userId": "int32",
                                     "movieId": "int32",
                                     "rating": "float32"
                                     })
        # unused
        df_tags = pd.read_csv(input_filepath_tags, sep=",",
                              usecols=["userId", "movieId", "tag"],
                              dtype={
                                  "userId": "int32",
                                  "movieId": "int32",
                                  "tag": "string"
                                  })

        # Merging datasets
        logger.info("Merging datasets...")

        # new approach, instead of merging gnoeme-scores with ratings,
        # we'll create a vektor from the genome-scores for each movie
        # afterwards we create vektors for each user, and average their
        # ratings for the movies they rated
        if not os.path.exists(os.path.join(output_filepath, "user_matrix.csv")):
            generate_user_matrix(df_ratings, df_scores, output_filepath, logger)

        # df = pd.concat(merged_chunks, ignore_index=True)
        # df.to_csv("merged_ratings_scores.csv", index=False)


        # df = pd.merge(df_ratings, df_scores, on='movieId', how='left')

        # Drop rows with missing values in specific columns
        col_to_drop_lines = ["rating", "tagId", "relevance"]  # Ensure these columns are used for the matrix
        logger.info(f"Dropping rows with missing values in columns: {col_to_drop_lines}")
        df = df.dropna(subset=col_to_drop_lines, axis=0)

        # Create a matrix with userId as rows and all other variables as columns
        logger.info("Creating user-feature matrix...")
        movie_matrix = df.set_index('userId')

        # Save the movie matrix to a CSV file
        output_file = os.path.join(output_filepath, 'movies_matrix.csv')
        movie_matrix.to_csv(output_file)

        logger.info(f"User-feature matrix saved to {output_file}")

        # Flatten the multi-level columns for better readability
        movie_matrix.columns = ['_'.join(map(str, col)) for col in movie_matrix.columns]

        # Reset the index to make `userId` a column
        movie_matrix.reset_index(inplace=True)

        # Save the movie matrix to a CSV file
        output_file = os.path.join(output_filepath, 'movies_matrix.csv')
        movie_matrix.to_csv(output_file, index=False)

        logger.info(f"Movie matrix saved to {output_file}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing file: {e}")
    except KeyError as e:
        logger.error(f"Key error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def generate_user_matrix(df_ratings, df_scores, output_filepath, logger):
    movie_embeddings = df_scores.pivot(index="movieId", columns="tagId", values="relevance").fillna(0)

    user_vectors = []
    user_ids = []

    grouped = df_ratings.groupby("userId")

    for user_id, group in tqdm(grouped):
        rated_movies = group["movieId"].values
        common_movies = [mid for mid in rated_movies if mid in movie_embeddings.index]

        if len(common_movies) == 0:
            continue  # skip user if no matching embeddings

        vectors = movie_embeddings.loc[common_movies]
        # Optional: mit Bewertung gewichten
        # weights = group.set_index("movieId").loc[common_movies]["rating"].values
        # weighted_vectors = vectors.mul(weights, axis=0)
        user_vector = vectors.mean(axis=0)  # oder: weighted_vectors.sum(axis=0) / weights.sum()
        user_vectors.append(user_vector)
        user_ids.append(user_id)

    user_matrix = pd.DataFrame(user_vectors, index=user_ids)
    user_matrix.index.name = "userId"
    user_matrix.to_csv(os.path.join(output_filepath, "user_matrix.csv"))
    logger.info(f"User matrix saved to {os.path.join(output_filepath, 'user_matrix.csv')}")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    input_filepath = "data/raw"
    output_filepath = "data/processed"

    main(input_filepath, output_filepath)
