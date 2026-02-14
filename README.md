# AI-Driven E-Waste Classification with Multi-Cloud Deployment

This project implements a CNN-based e-waste classification system using ResNet-18 and deploys the trained model as a Flask API for real-time prediction. The system supports multi-cloud deployment for scalability and reliability.

## Features
- CNN (ResNet-18) for 10-class e-waste classification  
- Flask-based REST API for real-time inference  
- Multi-cloud support (load balancing / failover)  
- Evaluation with accuracy, confusion matrix, precision/recall/F1

Cloud 2 – Logger Service (cloud2_logger.py)
cloud2_logger.py implements the secondary cloud service in the multi-cloud architecture. It runs as an independent Flask API that receives prediction results from the primary cloud (Cloud 1) and logs them for monitoring and reliability purposes.

Key Points:
Acts as Cloud 2 in the multi-cloud setup
Receives prediction logs from Cloud 1 via REST API
Stores logs (timestamp, image name, prediction) in a CSV file
Supports monitoring, auditing, and reliability
Demonstrates multi-cloud separation of services (inference vs logging)

## Setup
```bash
pip install -r requirements.txt
python train_cnn.py
python cloud_app.py
python test_api.py
```

