Intelligent Fraud Detection System for Financial Transactions

This project implements an end-to-end machine learning–based fraud detection system using the IEEE-CIS Fraud Detection dataset.
It covers the full pipeline: data preparation, model training, evaluation, and API-based inference.

The goal is to demonstrate how an intelligent fraud detection system is built and deployed, not just how a model is trained.

1. Project Overview

Problem
Detect fraudulent financial transactions from highly imbalanced transaction data.

Approach

Supervised binary classification (isFraud)

Time-based train/validation split

Gradient-boosted decision trees (LightGBM)

Probability-based scoring + decision threshold

REST API for real-time inference

2. Project Structure
Fraud-Detection-System-For-Financial-Transactions/
│
├── src/
│   ├── data.py        # Data loading and merging
│   ├── features.py    # Feature engineering and time split
│   ├── train.py       # Model training
│   ├── evaluate.py    # Model evaluation and metrics
│   └── api.py         # FastAPI inference service
│
├── data/              # IEEE-CIS dataset files (not committed)
├── models/            # Trained model + metadata (generated locally)
├── requirements.txt
├── .gitignore
└── README.md

3. Environment Setup
3.1 Create and activate virtual environment
python -m venv venv


PowerShell

venv\Scripts\activate


Git Bash

source venv/Scripts/activate

3.2 Install dependencies
pip install -r requirements.txt

4. Dataset Setup (IEEE-CIS)

Create a Kaggle account

Download the IEEE-CIS Fraud Detection dataset

Place the following files into the data/ directory:

data/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv


⚠️ The dataset is not committed to GitHub due to size and licensing.

5. Model Training

Train the fraud detection model using historical transaction data:

python -m src.train


This will:

Merge transaction and identity tables

Perform basic feature engineering

Split data using a time-based split

Train a LightGBM classifier

Save the trained model and metadata to models/

6. Model Evaluation

Evaluate the trained model on the validation set:

python -m src.evaluate


Reported metrics:

PR-AUC (Average Precision) – primary metric for imbalanced fraud detection

ROC-AUC – ranking performance

7. Run the Fraud Detection API

Start the FastAPI inference service:

uvicorn src.api:app --reload


The API will be available at:

http://127.0.0.1:8000

API Documentation (Swagger UI)

Open in browser:

http://127.0.0.1:8000/docs

3.2 Install dependencies
pip install -r requirements.txt

4. Dataset Setup (IEEE-CIS)

Create a Kaggle account

Download the IEEE-CIS Fraud Detection dataset

Place the following files into the data/ directory:

data/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv


⚠️ The dataset is not committed to GitHub due to size and licensing.

5. Model Training

Train the fraud detection model using historical transaction data:

python -m src.train


This will:

Merge transaction and identity tables

Perform basic feature engineering

Split data using a time-based split

Train a LightGBM classifier

Save the trained model and metadata to models/

6. Model Evaluation

Evaluate the trained model on the validation set:

python -m src.evaluate


Reported metrics:

PR-AUC (Average Precision) – primary metric for imbalanced fraud detection

ROC-AUC – ranking performance

7. Run the Fraud Detection API

Start the FastAPI inference service:

uvicorn src.api:app --reload


The API will be available at:

http://127.0.0.1:8000

API Documentation (Swagger UI)

Open in browser:

http://127.0.0.1:8000/docs