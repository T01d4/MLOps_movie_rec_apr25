#!/bin/bash
set -e

# This script prepares the environment and launches Docker Compose for the project.
# It is designed for Linux/macOS (bash). Use start.bat for Windows.

# Step 1: Create Python virtual environment if it does not exist
echo "Checking virtual environment..."
if [ ! -d .venv ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# Step 2: Copy dummy.env to .env if .env does not exist
if [ ! -f .env ]; then
    echo "Copying dummy.env to .env..."
    cp dummy.env .env
else
    echo ".env already exists, skipping copy."
fi

# Step 3: Revert any changes to dummy.env (prevents credentials from being pushed to git)
if [ -f dummy.env ]; then
    echo "Reverting changes to dummy.env..."
    git checkout -- dummy.env
else
    echo "dummy.env does not exist, skipping revert."
fi

# Step 4: Fix permissions for all important runtime directories (e.g. logs, models, data)
echo "Fixing permissions for runtime directories (ignore errors if not permitted)..."
# List all directories where containers write data:
RUNTIME_DIRS=(
    "./src/airflow/logs"
    "./models"
    "./data"
    "./reports"
    "./src/prometheus/data"
)

for DIR in "${RUNTIME_DIRS[@]}"
do
    if [ -d "$DIR" ]; then
        echo "Setting permissions for $DIR"
        sudo chown -R $(id -u):$(id -g) "$DIR" || true
        sudo chmod -R 775 "$DIR" || true
    else
        echo "$DIR does not exist, skipping."
    fi
done

echo "Starting Docker Compose..."
docker compose up --build -d

echo "Done."
