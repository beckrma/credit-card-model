"""Demo API for Credit Card Fraud Detection"""
import logging
import random
import sys
import time
import subprocess
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:

        # check if model is already trained and saved, if not, run training script
        model_path = PARENT_DIR / "models" / "autoencoder_model.pth"
        if not model_path.exists():
            # first, run training script to ensure model is trained and saved
            subprocess.run([sys.executable, "-m", "training.autoencoder_training"], cwd=PARENT_DIR, check=True)

        # then, check if synthetic data is already generated, if not, run synthetic_data_generation script
        synthetic_data_path = PARENT_DIR / "data" / "synthetic_creditcard_demo.csv"
        if not synthetic_data_path.exists():
            # run synthetic data generation script
            logging.info("Generating synthetic data for demo...")
            subprocess.run([sys.executable, "-m", "data.synthetic_data_generation"], cwd=PARENT_DIR, input="1\n", text=True, check=True, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)

        # then, start the FastAPI server to serve the prediction endpoint (without blocking main thread)
        logging.info("Starting FastAPI server...")
        api_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8001"], cwd=PARENT_DIR)

        # wait for fastapi server to start up before starting producer and consumer scripts
        time.sleep(5)

        # generate a random topic name for this demo
        topic_name = f"creditcard_topic_{random.randint(1000, 9999)}"

        # start producer script to send data to Kafka topic for consumer
        logging.info("Starting producer script...")
        subprocess.Popen([sys.executable, "-m", "demo.producer", topic_name], cwd=PARENT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # finally, run the consumer script to consume data from the Kafka topic and make predictions using the API
        logging.info("Starting consumer script...")
        subprocess.Popen([sys.executable, "-m", "demo.consumer", topic_name], cwd=PARENT_DIR)
    except subprocess.CalledProcessError as e:
        logging.error(e.stderr)

if __name__ == "__main__":
    main()

