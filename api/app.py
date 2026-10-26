from fastapi import FastAPI
from api.routes.prediction import router as predict_router
from api.routes.health import router as health_router

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Credit Card Fraud Detection API is running. Use the /predict endpoint to make predictions."}

app.include_router(predict_router)
app.include_router(health_router)