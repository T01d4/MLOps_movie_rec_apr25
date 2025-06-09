MovieReco - MLOPS Project
==============================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.


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
AIRFLOW__CORE__FERNET_KEY=your_fernet_key_here
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
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
DAGSHUB_USER=your_dagshub_username
DAGSHUB_TOKEN=your_dagshub_token

# --- Streamlit & API ---
MODEL_PATH=/app/models/model.pkl
JWT_SECRET=supersecretkey
API_URL=http://airflow-webserver:8080
TMDB_API_KEY=your_api_key_for_movie_pictures
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

---

### 5. Verify Airflow DAGs

1. Open the Airflow Web UI.
2. Check if the DAGs (e.g., `movie_recommendation_pipeline`) are listed.
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
