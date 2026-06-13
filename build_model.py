import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pickle.load(open("movies.pkl","rb"))

cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)

print("Model Ready")