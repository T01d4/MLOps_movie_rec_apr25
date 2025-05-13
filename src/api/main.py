from fastapi import FastAPI
from .routes import api_router

app = FastAPI()

# Include the routes
app.include_router(api_router)

@app.get("/")
def read_root():
    return {"data": "Hello World"}
