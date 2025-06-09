Project Name
==============================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.

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

# --- Airflow ---
AIRFLOW__CORE__EXECUTOR=LocalExecutor

AIRFLOW__CORE__FERNET_KEY=your_fernet_key_here  # stays like this

AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False

AIRFLOW__CORE__LOAD_EXAMPLES=False

AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow

AIRFLOW__WEBSERVER__SECRET_KEY=my_super_secret_key_42

PYTHONPATH=/opt/airflow/src

AIRFLOW_UID=50000

AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth,airflow.api.auth.backend.session

AIRFLOW_API_URL=http://airflow-webserver:8080/api/v1

# --- MLflow & DagsHub ---
MLFLOW_TRACKING_URI=https://dagshub.com/sacer11/MLOps_movie_rec_apr25.mlflow

MLFLOW_TRACKING_USERNAME=your_dagshub_username  # replace with your dagshub username

MLFLOW_TRACKING_PASSWORD=your_dagshub_token  # replace with your dagshub token

DAGSHUB_USER=your_dagshub_username  # example: yourusername

DAGSHUB_TOKEN=your_dagshub_token  # example: asdfgxcgvsedfrsdg (supposed to have 40 charakters)

# --- Streamlit & API ---
MODEL_PATH=/app/models/model.pkl

JWT_SECRET=supersecretkey

API_URL=http://api_service:8000

TMDB_API_KEY=your API Key for Movie Pictures

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
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── logs               <- Logs from training and predicting
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
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
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

### 1- Create a virtual environment using Virtualenv.

    `python -m venv my_env`

###   Activate it 

    `./my_env/Scripts/activate`

###   Install the packages from requirements.txt  (You can ignore the warning with "setup.py")

    `pip install -r .\requirements.txt`

### 2- Execute import_raw_data.py to import the 4 datasets (say yes when it asks you to create a new folder)

    `python .\src\data\import_raw_data.py` 

### 3- Execute make_dataset.py initializing `./data/raw` as input file path and `./data/processed` as output file path.

    `python .\src\data\make_dataset.py`

### 4- Execute build_features.py to preprocess the data (this can take a while)

    `python .\src\features\build_features.py`

### 5- Execute train_model.py to train the model

    `python .\src\models\train_model.py`

### 5- Finally, execute predict_model.py file to make the predictions (by default you will be printed predictions for the first 5 users of the dataset). 

    `python .\src\models\predict_model.py`

### Note that we have 10 recommandations per user

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
