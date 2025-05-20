starten





airflow:
docker compose -f airflow/docker-compose.airflow.yml down

docker compose -f airflow/docker-compose.airflow.yml up --build
http://localhost:8080/home

dann in neuer powersehll

streamlit:
docker compose -f streamlit_app/docker-compose.airflow.yml down

docker compose -f streamlit_app/docker-compose.streamlit.yml up --build
http://localhost:8501/


bei änderungen:
docker compose -f docker-compose.airflow.yml up airflow-init

docker compose -f docker-compose.airflow.yml build

docker compose -f docker-compose.airflow.yml up -d


