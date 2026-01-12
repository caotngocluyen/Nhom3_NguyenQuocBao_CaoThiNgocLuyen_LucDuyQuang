import os #ghép đường dẫn file ảnh
import numpy as np #thao tác mảng số
import tensorflow as tf # framework deep learning
from tensorflow.keras import layers, models #xây model, train, predict
from tensorflow.keras.applications import ResNet50

from .preprocessing import (
    load_and_split_data,
    IMG_SIZE_RESNET_EFF,
    SEED,
    save_history
)
from .feature import get_train_generator_resnet50, get_val_generator_resnet50

def build_resnet50_model(num_classes):
    #Load ResNet50 pretrained + đóng băng (Feature Extraction)
    base_model = ResNet50(
        weights="imagenet", #lấy trọng số học sẵn trên ImageNet
        include_top=False,#bỏ phần head gốc của ResNet50 (1000 lớp ImageNet)
        input_shape=(*IMG_SIZE_RESNET_EFF, 3)
    )
    base_model.trainable = False  # đóng băng, chỉ train phần head mình thêm, không cập nhật ResNet50

    #Gắn head mới
    #Nhận ảnh và biến ảnh thành feature map
    # Feature map Là các đặc trưng trừu tượng: cạnh, góc, texture, hình dạng,mẫu phức tạp (mắt, cánh, tổn thương…)
    inputs = layers.Input(shape=(*IMG_SIZE_RESNET_EFF, 3))
    # BN có 2 kiểu hành vi:
    # Khi training: dùng thống kê mean/var của batch hiện tại
    # Khi inference: dùng thống kê running mean/var đã lưu
    # Khi freeze backbone, thường muốn BN chạy như inference (ổn định), nên để training=False.
    x = base_model(inputs, training=False)          # training=False dùng thống kê batch hiện tại, giúp BatchNorm ổn định khi freeze

    # ResNet50 (include_top=False) trả ra feature map dạng (H, W, C), ví dụ: (7, 7, 2048)
    # C=2048 kênh: mỗi kênh như một “bộ dò đặc trưng”.
    # GlobalAveragePooling2D sẽ:
    # Lấy trung bình trên toàn bộ H×W cho từng kênh (7,7,2048) ==> (2048)
    # Tức là biến “bản đồ đặc trưng” thành vector đặc trưng gọn.
    x = layers.GlobalAveragePooling2D()(x) # Mỗi feature map = máy dò 1 đặc trưng (ví dụ: viền, đốm, kết cấu…), Nó lấy trung bình của MỖI feature map

    x = layers.BatchNormalization()(x)  #Chuẩn hoá các giá trị trong vector đặc trưng để chúng có phân phối ổn định hơn

    # Trong lúc train:random tắt 40% neuron, mỗi batch tắt neuron khác nhau
    # Tránh model học thuộc lòng. Bắt model học đặc trưng tổng quát hơn
    x = layers.Dropout(0.4)(x)

    #softmax biến 3 số đó thành xác suất (tổng = 1)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)#Tạo model với đầu vào là inputs và đầu ra là outputs
    return model, base_model

def get_callbacks_resnet50():
    #những thứ tự động can thiệp trong lúc train
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint( #Tự động lưu model tốt nhất trong lúc train
            "models/resnet50_best2.keras", #File model tốt nhất (có thể load lại sau)
            monitor="val_accuracy", #Theo dõi accuracy trên validation
            save_best_only=True, #Chỉ lưu khi val_accuracy tăng
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping( #Nếu model không còn tiến bộ nữa dừng train
            monitor="val_accuracy",
            patience=3, #Nếu 3 epoch liên tiếp mà val_accuracy không tăng dừng training
            restore_best_weights=True #Sau khi dừng ==> quay lại trọng số tốt nhất
        ),
        #Model học tốt lúc đầu, sau đó val_loss không giảm nữa, có thể do learning rate quá lớn
        tf.keras.callbacks.ReduceLROnPlateau( #giảm LR
            monitor="val_loss",
            factor=0.5, #Learning rate mới = learning rate cũ × 0.5
            patience=2,
            min_lr=1e-6,  # chặn LR giảm quá thấp
            verbose=1
        )
    ]
    return callbacks

def train_resnet50(EPOCHS_FE=10, EPOCHS_FT=20):
    tf.random.set_seed(SEED) #cố định tính ngẫu nhiên (random) của TensorFlow (Khởi tạo trọng số ban đầu của mạng, tắt neuron, xoay, lật ảnh)
    np.random.seed(SEED) #Cố định random của NumPy (Chia train/va, Tạo số ngẫu nhiên)

    train_df, val_df = load_and_split_data()
    train_gen = get_train_generator_resnet50(train_df)
    val_gen = get_val_generator_resnet50(val_df)

    # Lấy tên các lớp
    NUM_CLASSES = len(train_gen.class_indices)
    print("Classes:", train_gen.class_indices)
    print("NUM_CLASSES:", NUM_CLASSES)

    model, base_model = build_resnet50_model(NUM_CLASSES)
    model.summary() #in ra kiến trúc: số layer, shape từng tầng, số tham số.

    callbacks = get_callbacks_resnet50()

    # ======================
    # Feature Extraction
    # ======================
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3), #thuật toán cập nhật trọng số, sau mỗi lần model đoán sai điều chỉnh trọng số để đoán đúng hơn lần sau
        loss="categorical_crossentropy", #Là thước đo độ sai, Model cố gắng giảm loss xuống thấp nhất
        metrics=["accuracy"]
    )

    history_fe = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FE,
        callbacks=callbacks,
    )

    save_history(history_fe, "results/history/resnet50_history_fe.json")

    # ======================
    # Fine-tuning
    # ======================
    #Fine-tuning = mở một phần ResNet50 để học tinh chỉnh đặc trưng cho bài toán
    base_model.trainable = True #Cho phép ResNet50 có thể được cập nhật trọng số

    # Chỉ mở 30 layer cuối
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    #giảm learning rate xuống 1e-5 do:
    # Feature Extraction: Train từ đầu head cần học nhanh
    # Fine-tuning: Chỉ chỉnh nhẹ các layer đã học sẵn
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FT,
        callbacks=callbacks,
    )

    save_history(history_ft, "results/history/resnet50_history_ft.json")

    #lưu TOÀN BỘ mô hình vào file và Load lại mô hình đã lưu
    os.makedirs("models", exist_ok=True)
    model.save("models/resnet50_final2.keras")

    return model, history_fe, history_ft, val_gen

if __name__ == "__main__":
    # chạy: python -m src.model_resnet50
    model, history_fe, history_ft, val_gen = train_resnet50(EPOCHS_FE=10, EPOCHS_FT=20)

    # Sau khi train xong gọi evaluation để vẽ + lưu hình
    from .evaluation import plot_history_from_json, evaluate_confusion_report
    import json

    with open("results/history/resnet50_history_fe.json", "r", encoding="utf-8") as f:
        hfe = json.load(f)
    with open("results/history/resnet50_history_ft.json", "r", encoding="utf-8") as f:
        hft = json.load(f)

    plot_history_from_json(hfe, hft, title_prefix="ResNet50",
                          save_path="results/figures/resnet50_history.png")
    evaluate_confusion_report(model, val_gen, save_dir="results/figures", prefix="resnet50")
