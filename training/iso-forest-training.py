import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

def train_isolation_forest(training_data, contamination=0.0017, random_state=42, n_estimators=500):
    model = IsolationForest(contamination=contamination, random_state=random_state  , n_estimators=n_estimators)
    model.fit(training_data)
    return model

if __name__ == "__main__":
    df = pd.read_csv("data/creditcard.csv")

    features = df.drop (columns=["Class"])
    target = df["Class"]

    model = train_isolation_forest(features)
    predictions = model.predict(features)

    predictions = (predictions == -1).astype(int)

    print(classification_report(target, predictions))
    # autoencoder showed much better performance than isolated forest, but kept it here for documentation purposes