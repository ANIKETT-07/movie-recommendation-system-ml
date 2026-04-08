import streamlit as st
st.markdown("### 🔥 Get movie recommendations instantly!")
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("tmdb_5000_movies.csv")

# Preprocess
df = df[['title', 'overview', 'genres']].dropna()

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return " ".join(L)

df['genres'] = df['genres'].apply(convert)
df['combined'] = df['overview'] + " " + df['genres']

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
matrix = cv.fit_transform(df['combined'])

# Similarity
similarity = cosine_similarity(matrix)

df['title_lower'] = df['title'].str.lower()

# Recommendation function
def recommend(movie):
    movie = movie.lower()
    if movie not in df['title_lower'].values:
        return ["Movie not found"]

    idx = df[df['title_lower'] == movie].index[0]
    distances = similarity[idx]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    return [df.iloc[i[0]].title for i in movies_list]

# UI
st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter movie name")

if st.button("Recommend"):
    results = recommend(movie_name)

    st.subheader("Recommended Movies:")
    for movie in results:
        st.write(movie)