import os
import streamlit as st #dùng Streamlit, đặt bí danh là st để gọi kiểu st.xxx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image #đọc ảnh từ file uploader hoặc file mẫu.
import tensorflow as tf

# Import các hàm tiền xử lý tương ứng cho từng mô hình
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

# CẤU HÌNH GIAO DIỆN
st.set_page_config(page_title="Hệ thống Chẩn đoán Bệnh Gà", layout="wide")
st.markdown("<h1 style='text-align: center;'>CHẨN ĐOÁN BỆNH GÀ</h1>", unsafe_allow_html=True)
st.markdown("---")

#  CẤU HÌNH ĐƯỜNG DẪN
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model nằm trong thư mục models/
MODEL_RESNET_PATH = os.path.join(BASE_DIR, "models", "resnet50_final2.keras")
MODEL_EFF_PATH = os.path.join(BASE_DIR, "models", "EfficienNetB0.keras")

# ảnh mẫu nằm trong data/samples/
SAMPLE_DIR = os.path.join(BASE_DIR, "data", "samples")

IMG_SIZE = (224, 224)

# khớp nhãn trong CSV/train của bạn (thường là "New Castle Disease")
LABELS = ["Coccidiosis", "Healthy", "New Castle Disease", "Salmonella"]

# XỬ LÝ LỖI CUSTOM OBJECTS
# Hàm giả lập để nạp được mô hình có sử dụng Focal Loss khi train
def focal_loss_fixed(y_true, y_pred):
    return y_pred

# NẠP MÔ HÌNH 
@st.cache_resource
def load_models():
    custom_dict = {"focal_loss_fixed": focal_loss_fixed}
    # compile=False giúp nạp mô hình nhanh hơn và bỏ qua các lỗi hàm loss khi dự đoán
    res_model = tf.keras.models.load_model(MODEL_RESNET_PATH, custom_objects=custom_dict, compile=False)
    eff_model = tf.keras.models.load_model(MODEL_EFF_PATH, custom_objects=custom_dict, compile=False)
    return res_model, eff_model

try:
    model_res, model_eff = load_models()
except Exception as e:
    st.error(f"Lỗi nạp mô hình: {e}")
    st.info("Kiểm tra lại tên file mô hình trong thư mục đã khớp với code chưa.")
    st.stop()  #dừng app để tránh gọi predict khi model chưa load

# SIDEBAR
st.sidebar.header("Dữ liệu đầu vào")
uploaded = st.sidebar.file_uploader("Upload ảnh gà cần chẩn đoán", type=["jpg", "jpeg", "png"])

# Tự động quét ảnh trong thư mục samples nếu có
sample_paths = []
if os.path.exists(SAMPLE_DIR):
    sample_paths = [os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR) 
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]

sample_choice = st.sidebar.selectbox("Hoặc chọn ảnh mẫu có sẵn", ["(không)"] + sample_paths)

# LOGIC DỰ ĐOÁN
def predict_model(img_pil, model_type):
    img_resized = img_pil.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_type == "resnet":
        x = resnet_preprocess(img_array)
        probs = model_res.predict(x, verbose=0)[0]
    else:
        x = eff_preprocess(img_array)
        probs = model_eff.predict(x, verbose=0)[0]
        
    idx = int(np.argmax(probs))
    return idx, probs

# HIỂN THỊ KẾT QUẢ
#Lấy ảnh từ upload hoặc ảnh mẫu
img = None
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
elif sample_choice != "(không)":
    img = Image.open(sample_choice).convert("RGB")

if img is not None:
    # Hiển thị ảnh gốc ở giữa
    # Cột trống ở hai bên sẽ ép cột giữa vào trung tâm màn hình
    left_co, cent_co, last_co = st.columns([1, 2, 1])
    
    #mọi lệnh st.xxx bên trong sẽ vẽ vào cột giữa.
    with cent_co:
        #tiêu đề nhỏ.
        st.subheader("Ảnh đang kiểm tra")
        st.image(img, use_container_width=True) #tự co giãn theo bề rộng container.
    
    # Chia 2 cột để so sánh 2 mô hình
    col1, col2 = st.columns(2)
    
    # Thực hiện dự đoán
    with st.spinner('Đang tính toán...'): #hiển thị vòng xoay loading trong lúc chạy
        idx_res, probs_res = predict_model(img, "resnet")
        idx_eff, probs_eff = predict_model(img, "eff")

    # Cột 1: ResNet50
    with col1:
        st.subheader("1. Kết quả ResNet50")
        #hiển thị dạng thẻ 3 phần
        st.metric(label="Chẩn đoán",
                   value=LABELS[idx_res], 
                   delta=f"{probs_res[idx_res]*100:.2f}%") #in % dự đoán
        
        fig1, ax1 = plt.subplots() #tạo “khung hình” fig và “trục vẽ” ax
        colors = ['gray'] * len(LABELS) #tạo mảng màu mặc định xám cho mọi cột.
        colors[idx_res] = 'skyblue' #tô nổi cột có xác suất cao nhất.
        ax1.bar(LABELS, probs_res, color=colors)
        plt.xticks(rotation=45)
        ax1.set_title("Xác suất từng lớp (ResNet50)")
        st.pyplot(fig1) #đưa hình matplotlib lên Streamlit.

    # Cột 2: EfficientNetB0
    with col2:
        st.subheader("2. Kết quả EfficientNetB0")
        st.metric(label="Chẩn đoán", value=LABELS[idx_eff], delta=f"{probs_eff[idx_eff]*100:.2f}%")
        
        fig2, ax2 = plt.subplots()
        colors = ['gray'] * len(LABELS)
        colors[idx_eff] = 'lightgreen'
        ax2.bar(LABELS, probs_eff, color=colors)
        plt.xticks(rotation=45)
        ax2.set_title("Xác suất từng lớp (EfficientNetB0)")
        st.pyplot(fig2)

    # Bảng so sánh chi tiết
    st.divider() #tạo đường kẻ phân cách đẹp
    st.subheader(" Bảng so sánh chi tiết (%)")
    df_compare = pd.DataFrame({
        "Loại Bệnh / Sức khỏe": LABELS,
        "ResNet50 (%)": [f"{p*100:.2f}" for p in probs_res],
        "EfficientNetB0 (%)": [f"{p*100:.2f}" for p in probs_eff]
    })
    st.table(df_compare)
