import json
import time
import pandas as pd
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# send data to kafka topic for consumer
df = pd.read_csv("data/synthetic_creditcard_demo.csv")
df.drop(columns=['Class'], inplace=True)  # drop the 'Class' column as its not needed for prediction
for _, row in df.iterrows():
    producer.send('creditcard_topic', row.to_dict())
    time.sleep(0.1) # simulate real-time data streaming by adding a delay between sending messages
    