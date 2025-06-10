MovieRecomm - MLOPS Project
==============================

⚠️ BREAKING CHANGE: Repo complete Rewrite (force-push, Rewrite, File-Struktur, .dvc & Data/Models)! 
No `git pull` to lokal Repos  – use gitclone 
`git clone <REPO_URL>` in a fresh new folder 
otherwise Merge-conflicts, History-Problems or wrong DVC-States!

Project Setup
==============================

As of 02.06.2025

You need a single .env file in the project root (see example below).

DagsHub Token is required for DVC and MLflow tracking.

Required .env (root directory)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
==============================


# Required .env (root directory) !!!!!
Add your information to the dummy.env. The .env will be created during start.sh.
You have to change:
* DAGSHUB_USER=your_dagshub_username  # replace with your dagshub username
* DAGSHUB_TOKEN=your_dagshub_token  # replace with your dagshub token

Optional:
* TMDB_API_KEY=your API Key for Movie Pictures  # replace with your tmdb api, if you have one

##https://www.themoviedb.org/settings/api


============================== 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

⚠️ Replace your_dagshub_username and your_dagshub_token with your own credentials!
Never commit real tokens to git!

DVC Configuration (.dvc/config.local or .dvc/config)

==============================

['remote "dagshub"']

    auth = basic

    user = your_dagshub_username

    password = your_dagshub_token

==============================

⚠️ Do not commit your .dvc/config.local with credentials! (Add it to .gitignore)


**| Start the App  - after you have build the .env  -> docker compose up --build |**


Streamlit: http://localhost:8501/


https://dagshub.com/sacer11/MLOps_movie_rec_apr25.mlflow


**| USER: admin / PW: admin  |**



Airflow Web UI: http://localhost:8080/


**| USER: admin / PW: admin  |**


Airflow API (for service-to-service): http://airflow-webserver:8080/api/v1


Notes
All services read environment variables from the root .env (see env_file in docker-compose.yml).

For team work: Every developer needs their own DagsHub token in their local .env.

Never share .env and .dvc/config.local with secrets in public repos.

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── best_model_path<- path inside the container to the best performing model
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── airflow
    │   ├── dags                        <- path inside the container to the best performing model
    │   ├── logs                        <- The final, canonical data sets for modeling.
    │   ├── plugins                     <- The final, canonical data sets for modeling.
    │   ├── airflow.cfg                 <- The final, canonical data sets for modeling.
    │   ├── airflow.db                  <- The final, canonical data sets for modeling.
    │   ├── Dockerfile.airflow          <- The final, canonical data sets for modeling.
    │   ├── requirements.airflow.txt    <- The final, canonical data sets for modeling.
    │   └── webserver_config.py
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── check_structure.py    
    │   │   ├── import_raw_data.py 
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py

--------

## Steps to follow 

Convention : All python scripts must be run from the root specifying the relative file path.

### 1- run the start.sh script

    ```bash
    ./start.sh
    ```

### 2- Proceed to the relating web resources:

    Streamlit: http://localhost:8501/
    Airflow Web UI: http://localhost:8080/

The rest is self-explaining.

### Note that we have 10 recommandations per user

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
