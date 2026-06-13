import streamlit as st
import pickle
import time

#pageconfig

st.set_page_config(
    page_title="MovieMind",
    page_icon="🎬",
    layout="wide"
)

#loading data

movies = pickle.load(open('movies.pkl', 'rb'))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(
    movies['tags']
).toarray()

similarity = cosine_similarity(vectors)

#rec func

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


# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.card {
    background-color: #1E2329;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    min-height: 120px;
    margin-bottom: 20px;
    box-shadow: 0px 0px 8px rgba(255,255,255,0.08);
}

.card:hover {
    transform: scale(1.03);
}

.movie-title {
    font-size: 18px;
    font-weight: bold;
    color: white;
}

.hero {
    text-align:center;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# HERO SECTION
# ==========================

st.markdown("""
<div class='hero'>
<h1 style='color:#E50914;font-size:60px;'>
🎬 CineMatch AI
</h1>

<h4 style='color:gray;'>
Personalized Movie Recommendations Powered by Machine Learning
</h4>
</div>
""", unsafe_allow_html=True)

# ==========================
# METRICS
# ==========================

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Movies", len(movies))

with m2:
    st.metric("Features", "5000")

with m3:
    st.metric("Recommendations", "8")

st.markdown("---")

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.title("About")

    st.write("""
    This project uses:

    - Content Based Filtering
    - Count Vectorization
    - Cosine Similarity
    - TMDB Movie Dataset
    """)

    st.success("Built by Maisha")

# ==========================
# MOVIE SELECTOR
# ==========================

selected_movie = st.selectbox(
    "🎥 Select a Movie",
    movies['title'].values
)

# ==========================
# BUTTON
# ==========================

col1, col2, col3 = st.columns([1,2,1])

with col2:

    recommend_btn = st.button(
        "🚀 Generate Recommendations",
        use_container_width=True
    )

# ==========================
# RECOMMENDATIONS
# ==========================

if recommend_btn:

    with st.spinner("Finding similar movies..."):
        time.sleep(1.5)

    recommendations = recommend(selected_movie)

    st.success("Recommendations Generated")

    st.markdown("## 🍿 Recommended For You")

    row1 = st.columns(4)
    row2 = st.columns(4)

    for i in range(4):

        with row1[i]:

            st.markdown(
                f"""
                <div class='card'>
                <div class='movie-title'>
                {recommendations[i]}
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    for i in range(4,8):

        with row2[i-4]:

            st.markdown(
                f"""
                <div class='card'>
                <div class='movie-title'>
                {recommendations[i]}
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")

st.caption(
    "Built using Streamlit • Machine Learning • Cosine Similarity"
)