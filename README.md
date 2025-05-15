Movie Recommandation
====================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.

Project Organization
--------------------

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
    │   ├── api            <- Scripts for the api supposed to be running in the backgroud
    │   │   ├── main.py
    │   │   └── routes.py
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
    │
    │
    ├── tests              <- Source code for the tests of this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── api            <- Scripts for the api supposed to be running in the backgroud
    │   │   ├── main.py
    │   │   └── routes.py
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── test_check_structure.py    
    │   │   ├── test_import_raw_data.py 
    │   │   └── test_make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── test_build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── test_predict_model.py
    │   │   └── test_train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── test_visualize.py

--------

Scheme of containerization
--------------------------
```mermaid
graph TD
  %% External source
  S3[(S3 Bucket / External Data Source)]

  %% Orchestrator
  ORCH[Orchestrator (e.g. Prefect, Airflow, Makefile)]

  %% Pipeline Steps
  INGEST[data-ingest container]
  PREP[preprocess container]
  TRAIN[train-model container]
  API[predict-api container]
  UI[Streamlit UI (optional)]

  %% Volumes
  V1[/data volume/]
  V2[/processed volume/]
  V3[/models volume/]

  %% API endpoint
  API_OUT[(REST API endpoint)]

  %% Workflow
  S3 -->|Download| INGEST
  ORCH --> INGEST
  INGEST -->|writes raw data| V1
  V1 --> PREP
  ORCH --> PREP
  PREP -->|writes features| V2
  V2 --> TRAIN
  ORCH --> TRAIN
  TRAIN -->|writes model.pkl| V3
  V3 --> API
  API --> API_OUT
  UI -->|requests| API_OUT

  %% Volumes grouping
  subgraph Shared Volumes
    V1
    V2
    V3
  end

  %% Styles
  style ORCH fill:#ffe4b5,stroke:#333,stroke-width:1px
  style INGEST fill:#cde,stroke:#000
  style PREP fill:#cde,stroke:#000
  style TRAIN fill:#cde,stroke:#000
  style API fill:#cde,stroke:#000
  style UI fill:#e7f5ff,stroke:#000
  style V1 fill:#fff3cd,stroke:#ccc
  style V2 fill:#fff3cd,stroke:#ccc
  style V3 fill:#fff3cd,stroke:#ccc
  style API_OUT fill:#d4edda,stroke:#000
```

## Steps to follow 

Convention : All python scripts must be run from the root specifying the relative file path.

### 1- Start the devcontainer

* clone the repo on a location you like
* in vscode open the folder of your repo
* reopen in Container
* everything should be setup correctly

### 2- Execute import_raw_data.py to import the 4 datasets

    `python .\src\data\import_raw_data.py` 

### 3- Execute make_dataset.py

    `python .\src\data\make_dataset.py`

### 4- Execute build_features.py to preprocess the data (this can take a while)

    `python .\src\features\build_features.py`

### 5- Execute train_model.py to train the model

    `python .\src\models\train_model.py`

### 6- Finally, execute predict_model.py file to make the predictions (by default you will be printed predictions for the first 5 users of the dataset). 

    `python .\src\models\predict_model.py`

### Note that we have 10 recommandations per user

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
