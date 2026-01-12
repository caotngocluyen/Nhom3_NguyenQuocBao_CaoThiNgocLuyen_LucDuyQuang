import os #ghép đường dẫn file ảnh
import numpy as np #thao tác mảng số
import pandas as pd #đọc CSV, thao tác bảng
import tensorflow as tf # framework deep learning
from sklearn.model_selection import train_test_split #chia dữ liệu train/validation

import json
import os

def save_history(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True) #lấy thư mục cha của file cần lưu, tạo thư mục nếu chưa có
    with open(path, "w", encoding="utf-8") as f: #Nếu file tồn tại ==> ghi đè. Nếu chưa tồn tại ==> tạo file mới
        json.dump(history.history, f, ##Chuyển dictionary Python thành JSON và ghi xuống file
                  ensure_ascii=False, 
                  indent=2) 
    print("Saved history to", path) #file json dễ đọc hơn.

# =====================
# CONFIG (Cấu hình)
# =====================
IMG_DIR   = r"D:/Hocmay/ga/Train"         #folder chứa ảnh
CSV_PATH  = r"D:/Hocmay/ga/train_data.csv" #file csv chứa tên ảnh + label

# ResNet50/EfficientNet chuẩn 224x224, CNN 150x150
IMG_SIZE_RESNET_EFF = (224, 224)
IMG_SIZE_CNN = (150, 150)

BATCH_SIZE = 32            # mỗi lần cập nhật trọng số, model nhìn 32 ảnh
SEED = 42                  # giúp kết quả chia train/val “ổn định” (lần chạy sau giống lần trước)

tf.random.set_seed(SEED) #cố định tính ngẫu nhiên (random) của TensorFlow (Khởi tạo trọng số ban đầu của mạng, tắt neuron, xoay, lật ảnh)
np.random.seed(SEED) #Cố định random của NumPy (Chia train/va, Tạo số ngẫu nhiên)
# Nếu ko set mỗi lần chạy lại code kết quả khác nhau

#thống nhất thứ tự label khi hiển thị
LIST_CLASSES = ["Coccidiosis", "Healthy", "New Castle Disease", "Salmonella"]

def load_dataframe():
    #Đọc CSV + tạo filepath + lọc ảnh thiếu
    df = pd.read_csv(CSV_PATH)

    # CSV có 2 cột: images, label
    #tạo cột filepath chứa đường dẫn ảnh
    df["filepath"] = df["images"].apply(lambda x: os.path.join(IMG_DIR, x))

    # Lọc ảnh bị thiếu (nếu có)
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)

    df["label"] = df["label"].astype(str).str.strip()  # xóa khoảng trắng thừa nếu có

    print(df.head())
    print("Tổng ảnh:", len(df))
    print("Số lớp:", df["label"].nunique()) #Đếm số lớp khác nhau
    print(df["label"].value_counts())
    return df

def load_and_split_data(test_size=0.2):
    #Hàm đọc file csv và chia tập dữ liệu.
    df = load_dataframe()

    #Chia dữ liệu cho tập train/test
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=df["label"]
    )
    #stratify giữ nguyên tỉ lệ các lớp (label) ở cả train và val
    print("Train:", len(train_df), "Val:", len(val_df))
    return train_df, val_df
