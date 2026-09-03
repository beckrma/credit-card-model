import joblib
import torch
import numpy as np

from models.autoencoder import AutoEncoder

model = AutoEncoder(input_dim=30)

model.load_state_dict(torch.load("models/autoencoder_model.pth", weights_only=True))

model.eval()

scaler = joblib.load("models/scaler.pkl")

def predict_fraud(transaction):
    transaction = transaction["transaction"]
    transaction = np.array(transaction).reshape(1, -1)  # reshape to 2D array

    # scale the transaction data
    transaction_scaled = scaler.transform(transaction)

    # convert to tensor
    transaction_tensor = torch.FloatTensor(transaction_scaled)

    # run transaction through autoencoder
    with torch.no_grad():
        val_reconstructed = model(transaction_tensor)

    # calculate reconstruction error
    error = torch.mean(
        (transaction_tensor - val_reconstructed ) ** 2 
    ).item()

    threshold = 2.883143663406372  # this threshold was determined during training

    # determine if transaction is fradulent based on reconstruction error
    is_fraud = error > threshold
    return {
        "fraud": is_fraud,
        "reconstruction_error": error
    }
