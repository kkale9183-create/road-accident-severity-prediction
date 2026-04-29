import streamlit as st
import pickle

movies = pickle.load(open('movies.pkl','rb'))
model = pickle.load(open('knn_model.pkl','rb'))
vectors = pickle.load(open('vectors.pkl','rb'))

st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")

movie_list = movies['title'].values
selected_movie = st.selectbox("Select Movie", movie_list)

if st.button("Recommend"):
    index = movies[movies['title'] == selected_movie].index[0]
    distances, indices = model.kneighbors([vectors[index]])

    st.subheader("Recommended Movies:")
    for i in indices[0][1:]:
        st.write("👉 " + movies.iloc[i].title)
