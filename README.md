Fake News Detector is a machine learning-based web application that classifies news articles as Real or Fake using a pre-trained text classification model.

The application has been refactored into a cloud-native architecture consisting of:

Frontend: Static web interface served using Nginx
Backend: Flask REST API serving machine learning predictions
Containerization: Docker images for frontend and backend
Deployment Target: Kubernetes
Architecture
User Browser
      │
      ▼
Frontend (Nginx)
      │
      ▼
Flask API Backend
      │
      ▼
Machine Learning Model

The frontend collects user input and sends prediction requests to the backend API. The backend preprocesses the input, performs inference using the trained model, and returns the prediction.

Project Structure
Fake_News_Detector
├──App/
 ├── frontend/
 │   ├── templates/
 |       ├── index.html
 │   ├── static/
 |       ├── images (2)
 │    ├── Dockerfile
 │    └── .dockerignore
 |
 ├── backend/
 │   ├── app.py
 │   ├── requirements.txt
 │   ├── Dockerfile
 |   ├── .dockerignore
 │   └── models/
 │       ├── Fake_news_predictor.pkl
 │       ├── tfidf_text.pkl
 │       └── tfidf_title.pkl
 │
 └── README.md
Machine Learning Pipeline
Model
Trained binary classification model
Uses logistic regression
Serialized using Joblib
Stored as:
Fake_news_predictor.pkl
Feature Extraction

TF-IDF vectorization is used for:

News Title
News Content

Stored as:

tfidf_title.pkl
tfidf_text.pkl

The transformed feature vectors are combined and passed to the classifier for prediction.
![website_inputs](https://github.com/user-attachments/assets/34e99b4a-0490-40bf-bb29-09ed10d09115)
![website_output](https://github.com/user-attachments/assets/ae8647c7-b8db-4dcd-be98-eecda5db925d)

