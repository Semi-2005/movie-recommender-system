# Movie Recommendation System

ML-based movie recommendation system using content-based and collaborative filtering.

## Tech Stack
- Python
- FastAPI
- Scikit-learn
- React
- MovieLens Dataset

## Features
- Content-based recommendation
- Collaborative filtering
- REST API
- React UI

## How to run

### Backend
pip install -r requirements.txt
uvicorn backend.app.main:app --reload

## Data 
movie_features.csv is generated from raw MovieLens datasets using notebooks/eda.ipynb

### Frontend
cd frontend
npm run dev