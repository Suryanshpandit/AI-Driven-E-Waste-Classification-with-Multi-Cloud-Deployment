# AI-Driven E-Waste Classification with Multi-Cloud Deployment

This project implements a CNN-based e-waste classification system using ResNet-18 and deploys the trained model as a Flask API for real-time prediction. The system supports multi-cloud deployment for scalability and reliability.

## Features
- CNN (ResNet-18) for 10-class e-waste classification  
- Flask-based REST API for real-time inference  
- Multi-cloud support (load balancing / failover)  
- Evaluation with accuracy, confusion matrix, precision/recall/F1  

## Setup
```bash
pip install -r requirements.txt
python train_cnn.py
python cloud_app.py
python test_api.py
```

