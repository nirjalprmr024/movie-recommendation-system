import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# -------------------------------
# Load Data
# -------------------------------
movies = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None, names=range(24))
movies['title'] = movies[1].astype(str)
genre_cols = list(range(5, 24))
movies['combined'] = movies['title'] + " " + movies[genre_cols].astype(str).agg(' '.join, axis=1)

# -------------------------------
# Vectorize & Fit KNN
# -------------------------------
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies['combined'])
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(count_matrix)

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_knn_with_scores(movie_name, n_recommendations=5):
    matches = movies[movies['title'].str.contains(movie_name, case=False)]
    if len(matches) == 0:
        return [("Movie not found!", 0)]
    idx = matches.index[0]
    distances, indices = knn.kneighbors(count_matrix[idx], n_neighbors=n_recommendations+1)
    recommended_movies = [(movies.iloc[i]['title'], 1 - dist) for i, dist in zip(indices[0][1:], distances[0][1:])]
    return recommended_movies

# -------------------------------
# Streamlit Frontend
# -------------------------------
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie name and get similar movie recommendations!")

movie_input = st.text_input("Enter a movie name:")
num_rec = st.slider("Number of recommendations:", 1, 10, 5)

if movie_input:
    recommendations = recommend_knn_with_scores(movie_input, num_rec)
    if recommendations[0][0] == "Movie not found!":
        st.error("Movie not found. Please check spelling or try another movie.")
    else:
        st.subheader("Recommended Movies:")
        for rec, score in recommendations:
            st.write(f"✅ {rec} — Similarity Score: {score:.2f}")
