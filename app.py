import streamlit as st
import pickle
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#pageconfig

st.set_page_config(
    page_title="MovieMind",
    page_icon="🍿",
    layout="wide"
)

#load data

movies = pickle.load(open("movies.pkl", "rb"))

#similarity matrix

cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(
    movies['tags']
).toarray()

similarity = cosine_similarity(vectors)

#rec function

def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:9]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(
            movies.iloc[i[0]].title
        )

    return recommended_movies

#css

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.hero{
    text-align:center;
    padding:30px 0px;
}

.card{
    background: linear-gradient(145deg,#1f1f1f,#2a2a2a);
    padding:25px;
    border-radius:18px;
    text-align:center;
    min-height:160px;
    border:1px solid #333;
    transition:0.3s ease;
}

.card:hover{
    transform:translateY(-8px);
    border:1px solid #E50914;
    box-shadow:0px 10px 25px rgba(229,9,20,0.25);
}

.movie-title{
    color:white;
    font-size:20px;
    font-weight:600;
    margin-top:20px;
}

.stButton > button{

    background:#E50914;
    color:white;
    border:none;
    height:60px;
    border-radius:12px;
    font-size:20px;
    font-weight:bold;
    width:100%;
}

.stButton > button:hover{
    background:#ff1e2d;
}

</style>
""", unsafe_allow_html=True)

#sidebar

with st.sidebar:

    st.title("🍿 MovieMind")

    st.markdown("""
### About

MovieMind uses Content-Based Filtering and
Machine Learning to discover movies similar
to your favorites.

Select a movie and get personalized
recommendations instantly.
""")

    st.divider()

    st.caption("Built by Maisha")

#hero section

st.markdown("""
<div class='hero'>

<h1 style='font-size:72px;
font-weight:800;
margin-bottom:10px;'>

🍿 MovieMind

</h1>

<p style='font-size:24px;
color:#A0A0A0;'>

Find your next favorite movie in seconds

</p>

</div>
""", unsafe_allow_html=True)

st.markdown("---")

#movieselector

st.markdown("### 🎬 Select a Movie")

selected_movie = st.selectbox(
    "",
    movies['title'].values
)

#button

col1, col2, col3 = st.columns([1,2,1])

with col2:

    recommend_btn = st.button(
        "🚀 Get Recommendations"
    )

#results

if recommend_btn:

    with st.spinner(
        "🎬 Searching through thousands of movies..."
    ):
        time.sleep(1.5)

    recommendations = recommend(selected_movie)

    st.markdown("---")

    st.markdown(
        f"## Because you liked **{selected_movie}**"
    )

    st.write("")

    cols = st.columns(5)

    for idx, movie in enumerate(recommendations):

        with cols[idx]:

            st.markdown(
                f"""
                <div class='card'>

                <div style='font-size:45px;'>
                🎥
                </div>

                <div class='movie-title'>
                {movie}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")

st.caption(
    "Powered by Content-Based Filtering • Cosine Similarity • Streamlit"
)