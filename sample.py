import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("dataset/movies.csv", encoding="latin-1", sep="\t", usecols=["movie_id", "title", "genres"])
data_rating = pd.read_csv("dataset/ratings.csv", encoding="latin-1", sep="\t", usecols=["user_id", "movie_id", "rating", "timestamp", "user_emb_id", "movie_emb_id"])
data_user = pd.read_csv("dataset/users.csv", encoding="latin-1", sep="\t", usecols=["user_id", "gender", "age", "occupation", "zipcode", "age_desc", "occ_desc"])

data["genres"] = data["genres"].apply(lambda x: x.replace("|", " ").replace("-", ""))

tf = TfidfVectorizer(

)

tfidf_matrix = tf.fit_transform(data["genres"])
print(tfidf_matrix.shape)
print(tf.vocabulary_)

tfidf_matrix = pd.DataFrame(tfidf_matrix.todense(), columns=tf.get_feature_names_out(), index=data["title"])

cosine_results = pd.DataFrame(cosine_similarity(tfidf_matrix), index=data["title"], columns=data["title"])

top_k = 20
test_movie = "Toy Story (1995)"

def get_recommentations(title, df, num_movie=top_k):
    data = df.loc[title, :]
    data = data.sort_values(ascending=False)
    return data[:, num_movie].to_frame(name="score")

result = get_recommentations(test_movie, cosine_results)
