# Autoencoder Fraud Detector

Fintech backend focused project that focuses on building a fraud detection API that   
flags suspicious transactions in a simulated real time using Apache Kafka.

## 🚀 Features

- **Machine Learning Training:** In the training folder, there are two scripts that train an isolation forest, and an autoencoder. The isolation forest is in the folder just for documentation purposes, as the project is mainly implemented with the autoencoder model.
- **API:** REST API implemented with Python FastAPI library that connects to the autoencoder and returns Boolean true or false depending on if the transaction is fraud or not.
- **Demo:** Demo folder contains a script that simulates the entire process of this project, starting from training the autoencoder, synthesizing more data to test the model, and simulating real time credit card transactions using the synthesized data with Apache Kafka. 

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, ApacheKafka, SDV (Synthetic Data Vault)

## 🏁 Getting Started

Follow these steps to set up the project locally.


### Prerequisites

You need to have the following tools installed:
- Python version 3.11.1 or greater
- Pandas
- SDV
- Kafka-Python
- Pytorch
- FastAPI
- Git

***Important:***   
Make sure to download creditcard.csv into data folder for functionality!   
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 💡 Usage

Provide clear examples of how to run and use the project.

### Running the App
To start the development server, run:
```bash
cd credit-card-model
cd demo
python -m demo_api
```

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## ✉️ Contact

Ryan Beck - ryanbeck135@gmail.com
