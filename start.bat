@echo off
REM This script is intended for Windows environments.
REM For Linux/macOS, please use start.sh.

echo Checking virtual environment...
IF NOT EXIST .venv (
    echo Creating Python virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate

REM Copy dummy.env to .env if it doesn't exist
IF NOT EXIST .env (
    echo Copying dummy.env to .env...
    copy dummy.env .env
) ELSE (
    echo .env already exists, skipping copy.
)

REM Revert any changes to dummy.env to avoid pushing credentials to git
IF EXIST dummy.env (
    echo Reverting changes to dummy.env...
    git checkout -- dummy.env
) ELSE (
    echo dummy.env does not exist, skipping revert.
)

echo Starting Docker Compose...
docker compose up --build -d

echo Done.
