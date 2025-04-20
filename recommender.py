import pandas as pd
import joblib
import numpy as np

# Load data
df = pd.read_csv('data/processed_songs.csv')
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

feature_cols = [
    'valence', 'acousticness', 'danceability', 'duration_ms',
    'energy', 'instrumentalness', 'liveness', 'loudness',
    'speechiness', 'tempo'
]

def recommend(user_input):
    # Convert to array and scale
    input_scaled = scaler.transform([user_input])
    distances, indices = model.kneighbors(input_scaled)

    # Return song names and artists
    results = df.iloc[indices[0]][['name', 'artists']]
    return results
