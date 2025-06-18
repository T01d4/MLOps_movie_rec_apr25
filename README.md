MovieReco - MLOPS Project
==============================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.


otherwise Merge-conflicts, History-Problems or wrong DVC-States!

Project Setup

- Modelltrain & -versioning **MLflow**

- Taskmanager Workflows **Airflow**

- Deep Learning + Collaborative Filtering

- Monitoring  **Prometheus & Grafana**

- API & UI via **FastAPI + Streamlit**

- Container **Docker Compose**

-  Unit Tests & Linting via **GitHub Actions**
==============================

As of 02.06.2025

You need a single .env file in the project root (see example below).

DagsHub Token is required for DVC and MLflow tracking.

Required .env (root directory) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
==============================


# Required .env (root directory) !!!!!

# --- Airflow ---
AIRFLOW__CORE__EXECUTOR=LocalExecutor

AIRFLOW__CORE__FERNET_KEY=oLfPThKYdHr7hjymM4p97WLVlGzJByb9ULqat9MqObs=

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

PROM_URL=http://prometheus:9090



MLFLOW_TRACKING_USERNAME=your_dagshub_username

MLFLOW_TRACKING_PASSWORD=your_dagshub_token

API_URL=http://api_service:8000

GRAFANA_URL=http://localhost:3000/d/4f619286-f4fd-4040-821b-2189eb929e4c/movie-recommender?orgId=1&from=now-6h&to=now&timezone=browser

#https://www.themoviedb.org/settings/api

TMDB_API_KEY=48a501167bbb3fe39749d61d3fd3b0f6




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
    ├── .devcontainer/             # VS Code Umgebung
    ├── .github/                   # GitHub Actions Workflows
    ├── data
    │   ├── monitoring       <- Data from third party sources.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- HTML-Reports, Metriken, Prometheus Dumps
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src/                       # Source-code-
    │   ├── airflow/
    │   │   ├── Dockerfile.airflow
    │   │   ├── requirements.airflow.txt
    │   │   ├── webserver_config.py
    │   │   └── dags/
    │   │       ├── bento_api_pipeline.py
    │   │       ├── drift_monitoring_dag.py
    │   │       └── train_deep_model_dag.py
    │   ├── api_service/
    │   │   ├── Dockerfile.API
    │   │   ├── main.py
    │   │   ├── metrics.py
    │   │   ├── pw.py
    │   │   ├── recommend.py
    │   │   ├── trainer.py
    │   │   ├── requirements.api.txt
    │   │   └── users.json
    │   ├── Bento_service/
    │   │   ├── bento_service.py
    │   │   ├── Dockerfile.Bento
    │   │   ├── metrics.py
    │   │   └── requirements.bento.txt
    │   ├── monitoring/
    │   │   ├── analyze_drift.py
    │   │   ├── analyze_drift_requests.py
    │   │   ├── generate_drift_report_extended.py
    │   │   ├── generate_embedding.py
    │   │   └── plot_precision_history.py
    │   ├── movie/
    │   │   ├── __init__.py
    │   │   ├── data/
    │   │   │   ├── __init__.py
    │   │   │   ├── check_structure.py
    │   │   │   ├── import_raw_data.py
    │   │   │   └── make_dataset.py
    │   │   ├── features/
    │   │   │   ├── __init__.py
    │   │   │   └── build_features.py
    │   │   ├── models/
    │   │   │   ├── __init__.py
    │   │   │   ├── predict_best_model.py
    │   │   │   ├── predict_model.py
    │   │   │   ├── train_hybrid_deep_model.py
    │   │   │   ├── train_model.py
    │   │   │   └── validate_model.py
    │   │   └── visualization/
    │   │       ├── __init__.py
    │   │       └── visualize.py
    │   └── streamlit_app/
    │       ├── app.py
    │       ├── auth.py
    │       ├── recommender.py
    │       ├── training.py
    │       ├── requirements.streamlit.txt
    │       └── Dockerfile.streamlit
    ├── tests/                     # Unit Tests
    ├── docker-compose.yml         # Multi-Container Setup
    ├── setup.py                   # Python Paket
    ├── requirements.txt           # installers for dev container
    ├── .env                       # Lokale Variablen (nicht tracken)
--------

## CI / CD Pipeline

                                 ┌──────────────────────────────┐
                                 │        MLflow / DagsHub      │
                                 │     (Read-only: best_model)  │
                                 │     - For Predict Fast_API    │
                                 └────────────▲─────────────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              │     Prediction Service         │
                              │     (FastAPI /recommend)       │
                              └────▲──────────────┬────────────┘
                                   │              │
                                   │     ┌────────▼────────────┐
                                   │     │  Prometheus Export  │
                                   │     │  - Latency, User    │
                                   │     └─────────┬───────────┘
                                   │               │
   ┌───────────────────────┐              ┌─-───────▼──────────┐
   │      Streamlit        │─────────────►│ API EntryPoint     │
   │    Admin / User       │              │ (FastAPI)          │
   └──────────┬────────────┘              │ /auth /train ...   │
              │                           └─────────┬──────────┘
              ▼                                     ▼
 ┌──────────────────────────────┐        ┌────────────────────┐
 │        Admin Dashboard       │        │     Grafana Dash   │
 │ - Reports                    │        │     Metrics UI     │
 │ - Trigger DAGs               │        └────────────────────┘
 │ - train best_model           │
 │ - Metrics + Visualisation    │
 └────────────▲─────────────────┘
              │
              ▼
 ┌──────────────────────────────┐      ┌──────────────────────────────────────────────┐
 │          Airflow (DAGs)      │      │  MLflow / DagsHub (Tracking + Artefacts)     │
 │  - make_dataset              │─────►│  - Modell- and Metric-Logging                │
 │  - train_model               │      │  - Artefacts: Models, Reports, JSONs         │
 │  - validate_model            │      │  - Precision@10, Validation                  │
 │  - predict_model             │      └───────────────┬──────────────────────────────┘
 │  - drift_monitoring          │                      ▼
 └───────┬──────────────────────────────┬──────────────┐
         ▼                              ▼              │
 ┌──────────────────────────────┐   ┌────────────────────────────┐
 │   Evidently Drift Reports    │   │        Data Volume         │
 │   - HTML + JSON              │   │ RAW / Processed / Monitor  │
 │   - Saved to /data/reports   │   └────────────┬───────────────┘
 └──────────────┬───────────────┘                ▼
                └────────────────────────► Admin Dashboard
                                             ( Reports)
--------



## Steps to Execute the Project

### 1. Clone the Repository

If you haven't already cloned the repository, run:

```bash
git clone <REPO_URL>
```

---

### 2. Create the `.env` File

Create a `.env` file in the root directory:

```bash
touch .env
```

Edit the `.env` file and add the required environment variables:

```plaintext
# --- Airflow ---
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__FERNET_KEY=oLfPThKYdHr7hjymM4p97WLVlGzJByb9ULqat9MqObs=
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

PROM_URL=http://prometheus:9090

MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token

API_URL=http://api_service:8000

GRAFANA_URL=http://localhost:3000/d/4f619286-f4fd-4040-821b-2189eb929e4c/movie-recommender?orgId=1&from=now-6h&to=now&timezone=browser

#https://www.themoviedb.org/settings/api
TMDB_API_KEY=48a501167bbb3fe39749d61d3fd3b0f6
```

Replace placeholders like `your_fernet_key_here`, `your_dagshub_username`, etc., with your actual values.

---

### 3. Build and Start the Application

Run the following command to build and start all services:

```bash
docker compose up --build
```

---

### 4. Access the Services

Once the containers are running, access the services:

- **Streamlit App:**  
  URL: [http://localhost:8501](http://localhost:8501)  
  Login: `admin / admin`

- **Airflow Web UI:**  
  URL: [http://localhost:8080](http://localhost:8080)  
  Login: `admin / admin`

- **Grafna UI:**  see full link with for .ENV
  URL: http://localhost:3000/    
  Login: `admin / admin`

- **MLFlow Web UI:**  
  https://dagshub.com/sacer11/MLOps_movie_rec_apr25.mlflow


### 5. Verify Airflow DAGs

1. Open the Airflow Web UI.
2. Check if the DAGs (e.g., `deep_models_pipeline_pipeline`) are listed.
3. Trigger the DAG manually if needed.

---

### Optional: Run Python Scripts Locally

If you want to test the Python scripts manually (without Airflow), follow these steps:

#### Create Virtual Environment

```bash
python -m venv my_env
```

Activate the virtual environment:

```bash
source my_env/bin/activate  # For Linux/MacOS
./my_env/Scripts/activate   # For Windows
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Run Scripts

Execute the Python scripts step-by-step:

```bash
python src/data/import_raw_data.py
python src/data/make_dataset.py
python src/features/build_features.py
python src/models/train_model.py
python src/models/predict_model.py
```

---

Let me know if you need help with any specific command or step!

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
