from fastapi import APIRouter

api_router = APIRouter()

@api_router.get("/predict")
def get_prediction():
    return {"prediction": "this will be the prediction result"}
