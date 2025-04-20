import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib

# Load dataset
df = pd.read_csv('data/data.csv')

# Features to use
features = [
    'valence', 'acousticness', 'danceability', 'duration_ms',
    'energy', 'instrumentalness', 'liveness', 'loudness',
    'speechiness', 'tempo'
]

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN model
knn = NearestNeighbors(n_neighbors=10, algorithm='auto')
knn.fit(X_scaled)

# Save model and scaler
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
df[['name', 'artists'] + features].to_csv('data/processed_songs.csv', index=False)
