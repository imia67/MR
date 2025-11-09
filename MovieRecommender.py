import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("./movies (2).csv")
    ratings = pd.read_csv("./ratings.csv")
    movies["genres"] = movies["genres"].apply(lambda x: x.split("|")[0] if isinstance(x, str) else "Unknown")
    mergeddataset = pd.merge(ratings, movies, on="movieId", how="left")
    return movies, ratings, mergeddataset

movies, ratings, mergeddataset = load_data()

# -----------------------------
# BUILD CONTENT-BASED KNN MODEL
# -----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(movies["genres"].astype(str))

genre_knn = NearestNeighbors(metric="cosine")
genre_knn.fit(X)

def recommend_movie_by_title(title, n_neighbors=10):
    if title not in movies["title"].values:
        return ["Movie not found!"]
    indexval = movies[movies["title"] == title].index[0]
    distances, indices = genre_knn.kneighbors(X[indexval], n_neighbors=n_neighbors)
    recommendations = [movies.iloc[i]["title"] for i in indices[0][1:]]
    return recommendations

# -----------------------------
# BUILD USER-BASED KNN MODEL
# -----------------------------
usermovierel = mergeddataset.pivot_table(index="userId", columns="title", values="rating")
usermovierel.fillna(0, inplace=True)

user_knn = NearestNeighbors(metric="cosine", algorithm="brute")
user_knn.fit(usermovierel)

def recommend_by_user(userid, nneighbours=20, nrecommendations=10):
    if userid not in usermovierel.index:
        return ["User not found!"]
    uservector = usermovierel.loc[userid].values.reshape(1, -1)
    distances, indices = user_knn.kneighbors(uservector, n_neighbors=nneighbours + 1)
    similarusers = indices.flatten()[1:]
    similarusersratings = usermovierel.iloc[similarusers]
    meanrating = similarusersratings.mean(axis=0)
    userrating = usermovierel.loc[userid]
    meanrating = meanrating[userrating == 0]
    topmovies = meanrating.sort_values(ascending=False).head(nrecommendations)
    return list(topmovies.index)

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("🎥 Movie Recommendation System")
st.markdown("Built with **KNN** — both *Content-based* and *User-based Collaborative Filtering*!")

option = st.sidebar.radio("Choose Recommendation Type:", ["🎬 By Movie Title", "👤 By User ID"])

if option == "🎬 By Movie Title":
    st.header("Recommend Similar Movies (Genre-based)")
    movie_name = st.selectbox("Select a Movie:", movies["title"].sort_values().unique())
    if st.button("Recommend"):
        recs = recommend_movie_by_title(movie_name)
        st.subheader(f"Movies similar to **{movie_name}**:")
        for m in recs:
            st.write(f"- 🎞️ {m}")

elif option == "👤 By User ID":
    st.header("Recommend Movies Based on Similar Users")
    user_id = st.number_input("Enter User ID:", min_value=int(usermovierel.index.min()), max_value=int(usermovierel.index.max()), step=1)
    nrec = st.slider("Number of Recommendations:", 5, 20, 10)
    if st.button("Recommend"):
        recs = recommend_by_user(user_id, nrecommendations=nrec)
        st.subheader(f"Movies recommended for User {user_id}:")
        for m in recs:
            st.write(f"- 🍿 {m}")
