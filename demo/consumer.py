import json
import sys
import requests
import logging
import pandas as pd
from kafka import KafkaConsumer

MAX_MESSAGES = 10000
NUMBER_OF_FRAUD_TRANSACTIONS = 500
NUMBER_OF_FRAUD_FOUND = 0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
topic_name = sys.argv[1]
df = pd.read_csv("data/synthetic_creditcard_demo.csv")

consumer = KafkaConsumer(
    'creditcard_topic',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    group_id='creditcard_consumer_group',
    auto_offset_reset='earliest',
    enable_auto_commit=False
)

try:
    for index, message in enumerate(consumer, start=1):
        try:
            logging.info(f"Received message: {message.value}")

            transaction = message.value

            response = requests.post("http://localhost:8001/api/predict", json=transaction, timeout=(5, 30))
            logging.info(f"Prediction received: {response.json()}")
            prediction = response.json()        
            found_row = df[(df['Time'] == prediction['time']) & (df['Amount'] == prediction['amount'])]
            class_value = found_row.iloc[0]['Class']
            if prediction['fraud'] == True and class_value == 1:
                NUMBER_OF_FRAUD_FOUND += 1
            logging.info(f"Total number of transactions: {index}")
            logging.info(f"Total number of fraudulent transactions found so far: {NUMBER_OF_FRAUD_FOUND}")
            consumer.commit()
        except requests.RequestException:
            logging.exception("API request failed")
        except Exception as e:
            logging.error(f"Error processing message: {e}")
        if index >= MAX_MESSAGES:
            logging.info(f"Processed {MAX_MESSAGES} messages, stopping consumer.")
            break

except KeyboardInterrupt:
    logging.info("Consumer stopped.")
finally:
    consumer.close()
    logging.info(f"Total number of fraudulent transactions found: {NUMBER_OF_FRAUD_FOUND} out of {NUMBER_OF_FRAUD_TRANSACTIONS}")