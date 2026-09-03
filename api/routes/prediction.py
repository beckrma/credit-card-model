from fastapi import APIRouter
from services.prediction_service import predict_fraud

router = APIRouter(prefix="/api", tags=["Prediction"])


@router.post("/predict")
def predict(transaction: dict):
    return predict_fraud(transaction)
