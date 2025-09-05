import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime

# ===================================================================
# PHẦN 1: KẾT NỐI TỚI FIREBASE
# ===================================================================
# --- Khởi tạo Firebase Admin SDK ---
cred = credentials.Certificate("key/movie-recommendation-sys-f5230-firebase-adminsdk-fbsvc-a89cda2a14.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()
print("🚀 Kết nối tới Firestore thành công!")


# ===================================================================
# PHẦN 2: HÀM XÓA DỮ LIỆU CŨ TRONG COLLECTION
# ===================================================================
def delete_collection(coll_ref, batch_size):
    """Xóa một collection theo từng đợt (batch) để không bị quá tải."""
    docs = coll_ref.limit(batch_size).stream()
    deleted = 0

    for doc in docs:
        print(f"Đang xóa doc {doc.id}...")
        doc.reference.delete()
        deleted += 1

    if deleted >= batch_size:
        return delete_collection(coll_ref, batch_size)


# --- Chạy hàm xóa cho 3 collections ---
# print("\n🗑️ Bắt đầu xóa dữ liệu cũ...")
# collections_to_delete = ['movies', 'users', 'ratings']
# for coll_name in collections_to_delete:
#     coll_ref = db.collection(coll_name)
#     delete_collection(coll_ref, 50)  # Xóa 50 document mỗi lần
#     print(f"✅ Đã xóa xong collection: {coll_name}")
# print("👍 Xóa dữ liệu cũ hoàn tất!")

# ===================================================================
# PHẦN 3: XỬ LÝ VÀ TẢI LÊN DỮ LIỆU MỚI
# ===================================================================
print("\n✨ Bắt đầu xử lý và tải lên dữ liệu mới...")

# --- 3.1 MOVIES ---
print("Đang xử lý movies...")
movies_df = pd.read_csv(
    "dataset/movies.csv",
    encoding="latin-1",
    sep="\t",
    usecols=["movie_id", "title", "genres"]
)
movies_df['genres'] = movies_df['genres'].apply(lambda x: x.replace("-", "").split('|'))
movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)', expand=False)
movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
movies_df['title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip()

for movie in movies_df.to_dict('records'):
    if pd.isna(movie['year']): movie.pop('year', None)
    doc_ref = db.collection('movies').document(str(movie['movie_id']))
    doc_ref.set(movie)
print("✅ Đã tải xong dữ liệu movies!")

# --- 3.2 USERS ---
print("Đang xử lý users...")
users_df = pd.read_csv(
    "dataset/users.csv",
    encoding="latin-1",
    sep="\t",
    usecols=["user_id", "gender", "age", "occupation", "zipcode"]
)
users_df['age'] = pd.to_numeric(users_df['age'], errors='coerce').astype('Int64')
users_df['occupation'] = pd.to_numeric(users_df['occupation'], errors='coerce').astype('Int64')
users_df = users_df.dropna(subset=['age', 'occupation'])
for user in users_df.to_dict('records'):
    doc_ref = db.collection('users').document(str(user['user_id']))
    doc_ref.set(user)
print("✅ Đã tải xong dữ liệu users!")

# --- 3.3 RATINGS ---
print("Đang xử lý ratings...")
ratings_df = pd.read_csv(
    "dataset/ratings.csv",
    encoding="latin-1",
    sep="\t",
    usecols=["user_id", "movie_id", "rating", "timestamp"]
)
ratings_df['rating'] = pd.to_numeric(ratings_df['rating'], errors='coerce').astype(int)
ratings_df['timestamp'] = ratings_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))

batch = db.batch()
batch_counter = 0
batch_size = 499

total_ratings = len(ratings_df)
print(f"Tổng cộng có {total_ratings} ratings cần tải lên...")

for index, rating in enumerate(ratings_df.to_dict('records')):
    # Tạo một document mới trong collection 'ratings'
    doc_ref = db.collection('ratings').document()
    batch.set(doc_ref, rating)
    batch_counter += 1

    # Khi batch đủ lớn hoặc đã đến cuối file, thì commit (gửi) lên server
    if batch_counter >= batch_size or index == total_ratings - 1:
        print(f"Đang commit batch {index + 1}/{total_ratings}...")
        batch.commit()
        # Reset lại batch để bắt đầu lô mới
        batch = db.batch()
        batch_counter = 0

print("✅ Đã tải xong dữ liệu ratings!")


print("\n🎉 Hoàn tất! Dữ liệu mới đã được tải lên Firestore thành công.")
