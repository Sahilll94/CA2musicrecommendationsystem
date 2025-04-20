import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
knn = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load processed songs data
df = pd.read_csv('data/processed_songs.csv')

# List of features used for recommendation
features = [
    'valence', 'acousticness', 'danceability', 'duration_ms',
    'energy', 'instrumentalness', 'liveness', 'loudness',
    'speechiness', 'tempo'
]

# Streamlit App
st.title("🎵 Music Recommendation System")
st.markdown("Select your preferred audio features below to get song recommendations.")

# Collect user input for all features
user_input = []

for feature in features:
    min_val = df[feature].min()
    max_val = df[feature].max()
    label = f"{feature.capitalize()}"

    # Use slider and match types correctly
    if df[feature].dtype == 'int64':
        val = st.slider(label, int(min_val), int(max_val), int((min_val + max_val) / 2))
    else:
        val = st.slider(label, float(min_val), float(max_val), float((min_val + max_val) / 2), step=0.01)

    user_input.append(val)

# Convert to NumPy array and reshape
input_array = np.array(user_input).reshape(1, -1)

# Standardize user input
input_scaled = scaler.transform(input_array)

# Recommendation button
if st.button("Recommend Songs 🎧"):
    distances, indices = knn.kneighbors(input_scaled)

    st.subheader("🎶 Recommended Songs For You:")
    for idx in indices[0]:
        name = df.iloc[idx]['name']
        artist = df.iloc[idx]['artists']
        st.write(f"**{name}** by *{artist}*")



# Footer
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <div style="text-align: center; font-size: 16px; color: gray;">
        Made with ❤️ by <strong>Ishita Singh</strong> & <strong>Sakshi Shukla</strong>
    </div>
    """,
    unsafe_allow_html=True
)
