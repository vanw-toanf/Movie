import streamlit as st
import requests
import pandas as pd

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Hệ thống Gợi ý Phim",
    page_icon="🎬",
    layout="wide"
)

# --- GỌI API ---
API_URL = "http://127.0.0.1:8000"


@st.cache_data
def get_movie_list():
    """Lấy danh sách phim từ API để hiển thị trong selectbox."""
    try:
        response = requests.get(f"{API_URL}/movies")
        response.raise_for_status()  # Ném lỗi nếu request không thành công
        data = response.json()
        return data.get("movies", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi khi kết nối tới API: {e}")
        return []


# --- GIAO DIỆN ---
st.title("🎬 Hệ thống Gợi ý Phim")

# Lấy danh sách phim
movie_list = get_movie_list()

# Tạo 2 tab cho 2 chức năng
tab1, tab2 = st.tabs(["**Gợi ý theo Phim (Content-Based)**", "**Gợi ý cho Người dùng (Collaborative)**"])

# --- Tab 1: Content-Based Filtering ---
with tab1:
    st.header("Chọn một bộ phim bạn thích")

    if movie_list:
        selected_movie = st.selectbox(
            "Nhập hoặc chọn một bộ phim:",
            options=movie_list
        )

        if st.button("Tìm phim tương tự", key="content_based"):
            with st.spinner("Đang tìm kiếm..."):
                try:
                    res = requests.get(f"{API_URL}/recommendations/content-based/{selected_movie}")
                    res.raise_for_status()
                    recommendations = res.json().get("recommendations", [])

                    st.subheader(f"Các phim tương tự '{selected_movie}':")
                    if recommendations:
                        for movie in recommendations:
                            st.success(movie)
                    else:
                        st.warning("Không tìm thấy phim gợi ý.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Lỗi API: {e}")
    else:
        st.warning("Không thể tải danh sách phim từ API. Vui lòng kiểm tra lại backend.")

# --- Tab 2: Collaborative Filtering ---
with tab2:
    st.header("Nhập ID của bạn để nhận gợi ý")

    user_id = st.number_input("Nhập User ID:", min_value=1, step=1)

    if st.button("Gợi ý phim cho bạn", key="collaborative"):
        with st.spinner("Đang xử lý..."):
            try:
                res = requests.get(f"{API_URL}/recommendations/collaborative/{user_id}")
                res.raise_for_status()
                recommendations = res.json().get("recommendations", [])

                st.subheader(f"Gợi ý dành riêng cho User {user_id}:")
                if recommendations:
                    # Hiển thị kết quả trong 2 cột cho đẹp mắt
                    col1, col2 = st.columns(2)
                    for i, movie in enumerate(recommendations):
                        if i < 5:
                            with col1:
                                st.success(movie)
                        else:
                            with col2:
                                st.success(movie)
                else:
                    st.warning("Không tìm thấy phim gợi ý cho người dùng này.")
            except requests.exceptions.RequestException as e:
                st.error(f"Lỗi API: {e}")