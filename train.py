import os
import pickle
import pandas as pd

# Thư viện cho Content-Based
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Thư viện cho Collaborative Filtering
from surprise import SVD, Dataset, Reader
from surprise.dump import dump
from surprise.model_selection import cross_validate

# --- 1. CẤU HÌNH ---
# Đường dẫn tới các file dữ liệu và nơi lưu các mô hình đã train
DATA_DIR = "dataset"
ARTIFACTS_DIR = "artifacts"  # Thư mục để lưu các "sản phẩm" của mô hình

# Đường dẫn file input
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")

# Đường dẫn file output
# Lưu ma trận tương đồng của Content-Based model
COSINE_SIM_PATH = os.path.join(ARTIFACTS_DIR, "cosine_similarity.pkl")
# Lưu dataframe movies đã xử lý để tra cứu
MOVIES_DF_PATH = os.path.join(ARTIFACTS_DIR, "movies_df.pkl")
# Lưu Collaborative Filtering model
CF_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "svd_model.pkl")


def train_content_based():
    """
    Huấn luyện mô hình Content-Based Filtering.
    - Đọc file movies.csv
    - Xử lý cột 'genres'
    - Tính toán ma trận TF-IDF
    - Tính toán ma trận tương đồng Cosine
    - Lưu ma trận và dataframe movies vào file .pkl
    """
    print("--- 🚀 Bắt đầu training Content-Based model ---")

    # Đọc và xử lý dữ liệu
    df_movies = pd.read_csv(MOVIES_PATH, encoding="latin-1", sep="\t", usecols=["movie_id", "title", "genres"])
    # Xử lý cột genres để phù hợp cho TF-IDF
    df_movies['genres_processed'] = df_movies['genres'].str.replace('|', ' ', regex=False).str.replace('-', '', regex=False)

    # Khởi tạo và fit TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_movies['genres_processed'])

    # Tính ma trận tương đồng cosine
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Tạo DataFrame từ ma trận tương đồng để dễ tra cứu bằng title
    cosine_sim_df = pd.DataFrame(cosine_sim, index=df_movies['title'], columns=df_movies['title'])

    # Lưu các "sản phẩm" ra file
    print(f"Lưu ma trận tương đồng vào: {COSINE_SIM_PATH}")
    with open(COSINE_SIM_PATH, 'wb') as f:
        pickle.dump(cosine_sim_df, f)

    print(f"Lưu dataframe movies vào: {MOVIES_DF_PATH}")
    with open(MOVIES_DF_PATH, 'wb') as f:
        pickle.dump(df_movies, f)

    print("--- ✅ Training Content-Based hoàn tất. ---")


def train_collaborative_filtering():
    """
    Huấn luyện mô hình Collaborative Filtering bằng thư viện Surprise.
    - Đọc file ratings.csv
    - Load dữ liệu vào Dataset của Surprise
    - Huấn luyện thuật toán SVD
    - Lưu model đã huấn luyện ra file
    """
    print("\n--- 🚀 Bắt đầu training Collaborative Filtering model ---")

    # Đọc dữ liệu ratings
    df_ratings = pd.read_csv(RATINGS_PATH, encoding="latin-1", sep="\t", usecols=["user_id", "movie_id", "rating"])

    # Surprise yêu cầu một định dạng reader riêng
    reader = Reader(rating_scale=(1, 5))  # Thang điểm rating từ 1 đến 5

    # Load dataframe vào dataset của Surprise
    data = Dataset.load_from_df(df_ratings[['user_id', 'movie_id', 'rating']], reader)

    # Xây dựng trainset từ toàn bộ dữ liệu
    trainset = data.build_full_trainset()

    # Sử dụng thuật toán SVD (một trong những thuật toán phổ biến nhất)
    algo = SVD(n_factors=100, n_epochs=20, random_state=42, verbose=True)

    # Huấn luyện mô hình
    print("Đang huấn luyện mô hình SVD...")
    algo.fit(trainset)

    # Lưu model đã huấn luyện
    # Surprise có hàm dump riêng để tối ưu việc lưu model
    print(f"Lưu model SVD vào: {CF_MODEL_PATH}")
    dump(file_name=CF_MODEL_PATH, algo=algo)

    print("--- ✅ Training Collaborative Filtering hoàn tất. ---")

    print("=" * 50)

    print("\n--- 📊 Bắt đầu đánh giá mô hình SVD ---")
    # Chạy cross-validation với 5-folds
    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Lấy kết quả trung bình
    mean_rmse = cv_results['test_rmse'].mean()
    mean_mae = cv_results['test_mae'].mean()

    print(f"\nKết quả đánh giá trung bình:")
    print(f"RMSE: {mean_rmse:.4f}")
    print(f"MAE:  {mean_mae:.4f}")


if __name__ == "__main__":
    print("Bắt đầu quá trình training...")

    # Tạo thư mục 'artifacts' nếu chưa tồn tại
    if not os.path.exists(ARTIFACTS_DIR):
        print(f"Tạo thư mục {ARTIFACTS_DIR} để lưu model...")
        os.makedirs(ARTIFACTS_DIR)

    # Chạy training cho cả hai mô hình
    train_content_based()
    train_collaborative_filtering()

    print("\n🎉 Toàn bộ quá trình training đã hoàn tất thành công!")
    print(f"Các file model đã được lưu trong thư mục '{ARTIFACTS_DIR}'.")

    print("=" * 50)

