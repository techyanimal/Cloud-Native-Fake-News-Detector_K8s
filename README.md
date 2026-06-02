# Fake News Detector - Cloud Native Deployment

## Overview

Fake News Detector is a machine learning-based web application that classifies news articles as **Real** or **Fake** using a pre-trained text classification model.

The application has been refactored into a cloud-native architecture consisting of:

- **Frontend:** Static web interface served using Nginx
- **Backend:** Flask REST API serving machine learning predictions
- **Containerization:** Docker images for frontend and backend
- **Deployment Target:** Kubernetes

---

## Architecture (K8S service included)

```text
User Browser
      │
      ▼
Frontend service
      │
      ▼
Frontend pod (Nginx)
      │
      ▼
Backend service (:5000)
      │
      ▼
Backend pod (Flask API)
      │
      ▼
Machine Learning Model
```

The frontend collects user input and sends prediction requests to the backend API. The backend preprocesses the input, performs inference using the trained model, and returns the prediction.

---

## Project Structure

```text
Fake_News_Detector/
├── App/
  ├── frontend/
  │   ├── templates/
  |       ├── index.html
  │   ├── static/
  |       ├── background.png
  |       ├── header-background.png
  |   ├── .dockerignore
  │   └── Dockerfile
  │ 
  ├── backend/
  │   ├── app.py
  │   ├── requirements.txt
  │   ├── Dockerfile
  |   ├── .dockerignore
  │   └── models/
  │       ├── Fake_news_predictor.pkl
  |       ├── tfidf_text.pkl
  │       └── tfidf_title.pkl
  │
├── Dataset/
  |   ├── fake.csv
  |   ├── true.csv
├── MLcode/
  |   ├── model.py
  |   ├── using_model.py
├── manifests/
  |   ├── backend-service.yaml
  |   ├── backend.yaml
  |   ├── frontend-service.yaml
  |   ├── frontend.yaml
└── README.md
```

---

## Machine Learning Pipeline

### Model

- Trained binary classification model
- Uses Logistic Regression
- Serialized using Joblib

Stored as:

```text
Fake_news_predictor.pkl
```

### Feature Extraction

TF-IDF vectorization is used for:

- News Title
- News Content

Stored as:

```text
tfidf_title.pkl
tfidf_text.pkl
```

The transformed feature vectors are combined and passed to the classifier for prediction.

---

## Nginx Configuration

The frontend is served using **Nginx**. In addition to serving static files, Nginx acts as a **reverse proxy** for API requests and forwards them to the Flask backend service running inside Kubernetes.

### Request Flow

```text
Browser
   │
   ▼
Frontend Service (Nginx)
   │
   ├── /           → Static HTML/CSS/JS
   │
   └── /predict    → Backend Service (Flask)
```

### Advantages of Reverse Proxying

- Eliminates hardcoded backend IP addresses and ports.
- Avoids browser CORS issues.
- Allows backend services to remain internal to the Kubernetes cluster.
- Simplifies frontend configuration across development and production environments.
- Enables seamless service discovery using Kubernetes Services.

---

## Kubernetes Deployment

### Deploy Backend

kubectl apply -f manifests/backend.yaml
kubectl apply -f manifests/backend-service.yaml

### Deploy Frontend

kubectl apply -f manifests/frontend.yaml
kubectl apply -f manifests/frontend-service.yaml

### Verify

kubectl get pods
kubectl get svc

### Access Frontend

minikube service frontend-service

---

## Website Interface (Initially)

![website_inputs](https://github.com/user-attachments/assets/34e99b4a-0490-40bf-bb29-09ed10d09115)
![website_output](https://github.com/user-attachments/assets/ae8647c7-b8db-4dcd-be98-eecda5db925d)
