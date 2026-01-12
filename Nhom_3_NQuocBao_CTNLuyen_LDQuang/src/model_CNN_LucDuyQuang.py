import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np #thao tác mảng số

from .preprocessing import load_and_split_data, SEED, save_history
from .feature import get_train_val_generators_cnn

def build_cnn_model(num_classes):
    # 4. Xây dựng mô hình CNN
    # Đây là một mạng CNN đơn giản dùng để phân loại ảnh hoa.
    # Mỗi lớp (layer) trong mô hình có một vai trò riêng biệt trong việc trích xuất đặc trưng (feature extraction) và phân loại (classification).
    model = models.Sequential([
        layers.Input(shape=(150,150,3)), #Xác định kích thước ảnh đầu vào: 150×150 pixel, 3 kênh màu (RGB).
        layers.Conv2D(32, (3,3), activation='relu'), #32: số bộ lọc (filters) = số lượng bản đồ đặc trưng (feature maps) được học.
        #(3,3): kích thước kernel (bộ lọc) 3×3 dùng để quét qua ảnh.activation='relu': hàm kích hoạt ReLU (Rectified Linear Unit) giúp loại bỏ giá trị âm
        #-->trích xuất đặc trưng cơ bản như cạnh, góc, màu sắc.
        layers.MaxPooling2D(2,2), #(2,2) có nghĩa là lấy giá trị lớn nhất trong vùng 2×2 --> giảm độ phân giải nhưng giữ lại đặc trưng quan trọng.

        layers.Conv2D(64, (3,3), activation='relu'), #Lớp tích chập thứ hai, với 64 filters -->học các đặc trưng phức tạp hơn như đường cong, họa tiết, hình dạng nhỏ.
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'), #Lớp tích chập thứ ba, có 128 filters → học đặc trưng ở mức trừu tượng cao hơn (như hình dạng cánh hoa, kết cấu cụ thể).
        layers.MaxPooling2D(2,2),

        layers.Flatten(), #Biến đầu ra 3D (chiều cao × chiều rộng × số kênh) thành một vector 1D.
        layers.Dense(128, activation='relu'), #Lớp Fully Connected (liên kết đầy đủ) với 128 neuron --> học mối quan hệ phi tuyến giữa các đặc trưng
        layers.Dropout(0.3),           # Dropout ngẫu nhiên loại bỏ 30% neuron trong lúc huấn luyện --tránh overfitting
        layers.Dense(num_classes, activation='softmax') #activation='softmax' biến đầu ra thành phân phối xác suất → tổng các xác suất = 1.
        # mô hình chọn lớp có xác suất cao nhất.
    ])
    return model

def train_cnn(EPOCHS=5):
    tf.random.set_seed(SEED) #cố định tính ngẫu nhiên (random) của TensorFlow (Khởi tạo trọng số ban đầu của mạng, tắt neuron, xoay, lật ảnh)
    np.random.seed(SEED) #Cố định random của NumPy (Chia train/va, Tạo số ngẫu nhiên)

    train_df, val_df = load_and_split_data()

    train_gen, val_gen = get_train_val_generators_cnn(train_df, val_split=0.2)

    # Lấy tên các lớp
    NUM_CLASSES = len(train_gen.class_indices)
    print("Classes:", train_gen.class_indices)
    print("NUM_CLASSES:", NUM_CLASSES)

    model = build_cnn_model(NUM_CLASSES)
    model.summary()

    # 5. Compile mô hình
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 6.Huấn luyện mô hình
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("results/history", exist_ok=True)

    save_history(history, "results/history/cnn_history.json")

    #lưu TOÀN BỘ mô hình vào file và Load lại mô hình đã lưu
    model.save("models/CNN_final.keras")

    return model, history, val_gen

if __name__ == "__main__":
    # chạy: python -m src.model_cnn
    model, history, val_gen = train_cnn(EPOCHS=5)

    from .evaluation import plot_history_from_json, evaluate_confusion_report
    import json

    with open("results/history/cnn_history.json", "r", encoding="utf-8") as f:
        h = json.load(f)

    plot_history_from_json(h, None, title_prefix="CNN",
                          save_path="results/figures/cnn_history.png")
    evaluate_confusion_report(model, val_gen, save_dir="results/figures", prefix="cnn")
