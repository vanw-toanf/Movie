import pickle
import pandas as pd
from fastapi import FastAPI
from surprise.dump import load as load_surprise_model

# --- 1. KHỞI TẠO ỨNG DỤNG FASTAPI ---
app = FastAPI(
    title="Movie Recommendation API",
    description="API phục vụ hai mô hình gợi ý: Content-Based và Collaborative Filtering."
)


# --- 2. TẢI CÁC MODEL VÀ DỮ LIỆU ĐÃ HUẤN LUYỆN ---
# Dùng @app.on_event("startup") để đảm bảo model chỉ được tải 1 lần khi server khởi động
@app.on_event("startup")
async def load_models():
    global movies_df, cosine_sim_df, svd_model, ratings_df
    print("🚀 Bắt đầu tải các model và dữ liệu...")

    # Tải các file cho Content-Based model
    with open("artifacts/movies_df.pkl", "rb") as f:
        movies_df = pickle.load(f)
    with open("artifacts/cosine_similarity.pkl", "rb") as f:
        cosine_sim_df = pickle.load(f)

    # Tải file cho Collaborative Filtering model
    _, svd_model = load_surprise_model("artifacts/svd_model.pkl")

    # Tải thêm file ratings để biết user đã xem phim nào
    ratings_df = pd.read_csv("dataset/ratings.csv", encoding="latin-1", sep="\t", usecols=["user_id", "movie_id"])

    print("✅ Tải model và dữ liệu thành công!")


# --- 3. ĐỊNH NGHĨA CÁC API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với API Gợi ý Phim!"}


@app.get("/movies")
def get_movie_list():
    """Lấy danh sách tất cả các phim để hiển thị trên giao diện."""
    return {"movies": movies_df['title'].tolist()}


@app.get("/recommendations/content-based/{movie_title}")
def get_content_based_recommendations(movie_title: str, top_k: int = 10):
    """
    Gợi ý phim dựa trên nội dung (Content-Based).
    Lấy ra top_k phim có độ tương đồng cao nhất với movie_title.
    """
    if movie_title not in cosine_sim_df.index:
        return {"error": "Không tìm thấy phim này."}

    # Lấy ra các phim tương đồng và sắp xếp
    similar_movies = cosine_sim_df[movie_title].sort_values(ascending=False)

    # Lấy top_k+1 (vì phim đầu tiên là chính nó) và bỏ phim đầu tiên
    top_movies = similar_movies.iloc[1:top_k + 1].index.tolist()

    return {"recommendations": top_movies}


@app.get("/recommendations/collaborative/{user_id}")
def get_collaborative_filtering_recommendations(user_id: int, top_k: int = 10):
    """
    Gợi ý phim cho người dùng (Collaborative Filtering).
    Dự đoán điểm rating cho những phim user chưa xem và trả về top_k phim có điểm dự đoán cao nhất.
    """
    # Lấy danh sách các movie_id mà user ĐÃ xem
    watched_movie_ids = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].unique()

    # Lấy tất cả movie_id mà user CHƯA xem
    unwatched_movie_ids = movies_df[~movies_df['movie_id'].isin(watched_movie_ids)]['movie_id'].tolist()

    # Dự đoán điểm rating cho các phim chưa xem
    predictions = [svd_model.predict(user_id, movie_id) for movie_id in unwatched_movie_ids]

    # Sắp xếp các dự đoán theo điểm số giảm dần
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Lấy ra top_k movie_id
    top_movie_ids = [pred.iid for pred in predictions[:top_k]]

    # Chuyển từ movie_id sang title
    top_movie_titles = movies_df[movies_df['movie_id'].isin(top_movie_ids)]['title'].tolist()

    return {"recommendations": top_movie_titles}