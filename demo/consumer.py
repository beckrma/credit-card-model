import json
import pandas as pd
import requests
from kafka import KafkaConsumer

NUMBER_OF_FRAUD_TRANSACTIONS = 500
NUMBER_OF_FRAUD_FOUND = 0

df = pd.read_csv("data/synthetic_creditcard_demo.csv")

consumer = KafkaConsumer(
    'creditcard_topic',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True
)

try:
    for message in consumer:
        transaction = message.value

        response = requests.post("http://localhost:8001/api/predict", json=transaction)
        if response.status_code == 200:
            prediction = response.json()        
            found_row = df[(df['Time'] == prediction['time']) & (df['Amount'] == prediction['amount'])]
            class_value = found_row.iloc[0]['Class']
            if prediction['fraud'] == True and class_value == 1:
                NUMBER_OF_FRAUD_FOUND += 1
        else:
            print(f"Failed to get prediction for transaction: {transaction}, Status Code: {response.status_code}")
except KeyboardInterrupt:
    print("Consumer stopped.")
finally:
    consumer.close()
    print(f"Total number of fraudulent transactions found: {NUMBER_OF_FRAUD_FOUND} out of {NUMBER_OF_FRAUD_TRANSACTIONS}")