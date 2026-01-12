import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils import class_weight

from .preprocessing import load_and_split_data, IMG_SIZE_RESNET_EFF, SEED, save_history
from .feature import get_train_generator_efficient, get_val_generator_efficient

tf.random.set_seed(SEED)
np.random.seed(SEED)

# Hàm focal_loss giúp mô hình tập trung vào các mẫu khó.
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred)) # Xác suất dự đoán đúng cho lớp thực tế
        return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) # Giảm trọng số của các mẫu dễ học, tăng trọng số mẫu khó
    return focal_loss_fixed

# Xây dựng kiến trúc mô hình
def build_model(num_classes):
    base_model = EfficientNetB0( # Sử dụng mô hình EfficientNetB0 đã được huấn luyện sẵn trên tập ImageNet
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE_RESNET_EFF, 3)
        )
    base_model.trainable = False # Đóng băng mô hình gốc để không làm hỏng kiến thức cũ trong giai đoạn đầu

    inputs = layers.Input(shape=(*IMG_SIZE_RESNET_EFF, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x) # Chuyển đổi dữ liệu từ không gian 2D sang vector phẳng
    x = layers.BatchNormalization()(x) # Chuẩn hóa dữ liệu giúp mô hình hội tụ nhanh hơn

    x = layers.Dense(512, activation='relu')(x) # Thêm các lớp Dense để học các đặc trưng riêng của phân gà
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x) # Ngẫu nhiên ngắt kết nối 40% nơ-ron để chống Overfitting

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x) # Ngẫu nhiên ngắt kết nối 30% nơ-ron để chống Overfitting

    outputs = layers.Dense(num_classes, activation="softmax")(x) # Lớp đầu ra với hàm Softmax để xuất xác suất cho từng loại bệnh
    return models.Model(inputs, outputs), base_model

def train_efficient(EPOCHS_FE=5, EPOCHS_FT=15):
    # Load Data
    train_df, val_df = load_and_split_data()
    train_gen = get_train_generator_efficient(train_df)
    val_gen = get_val_generator_efficient(val_df)

    # Khởi tạo model
    model, base_model = build_model(num_classes=len(train_gen.class_indices))

    # Tính Class Weights, xử lý mất cân bằng dữ liệu
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
    cw_dict = dict(enumerate(weights))
    cw_dict[3] *= 0.8  # Giảm Salmonella
    cw_dict[2] *= 1.2  # Tăng Newcastle

    os.makedirs("models", exist_ok=True)
    os.makedirs("results/history", exist_ok=True)

    # Giai đoạn 1
    callbacks_fe = [
        tf.keras.callbacks.ModelCheckpoint("models/EfficienNetB0.keras", monitor="val_loss", save_best_only=True, verbose=1), # Tự động lưu lại mô hình tốt nhất
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1) # Tự động dừng huấn luyện sớm (EarlyStopping)
    ]

    # Cấu hình mô hình
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05), # Label smoothing giúp mô hình bớt tự tin
        metrics=["accuracy"]
    )

    # Bắt đầu huấn luyện
    history_fe = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FE,
        callbacks=callbacks_fe
    )

    save_history(history_fe, "results/history/efficient_history_fe.json")

    # Giai đoạn 2
    # Fine-tuning
    base_model.trainable = True # Mở đóng băng toàn bộ
    for layer in base_model.layers[:-60]: # đóng băng lại trừ 60 lớp cuối cùng
        layer.trainable = False

    callbacks_ft = [
        tf.keras.callbacks.ModelCheckpoint("models/EfficienNetB0.keras", monitor="val_loss", save_best_only=True, verbose=1),  # Tự động lưu lại mô hình tốt nhất
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1), # Tự động dừng huấn luyện sớm (EarlyStopping)
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1) # Tự động giảm Learning Rate nếu mô hình không tiến triển
    ]

    #Cấu hình mô hình giai đoạn 2
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=focal_loss(),
        metrics=["accuracy"]
    )

    # Bắt đầu huấn luyện giai đoạn 2
    history_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FT,
        class_weight=cw_dict,
        callbacks=callbacks_ft
    )

    save_history(history_ft, "results/history/efficient_history_ft.json")

    # Lưu final (nếu bạn muốn tách best/final)
    model.save("models/efficient_final.keras")

    return model, history_fe, history_ft, val_gen

if __name__ == "__main__":
    # chạy: python -m src.model_efficient
    model, history_fe, history_ft, val_gen = train_efficient()

    from .evaluation import plot_history_from_json, evaluate_confusion_report, evaluate_with_threshold
    import json

    with open("results/history/efficient_history_fe.json", "r", encoding="utf-8") as f:
        hfe = json.load(f)
    with open("results/history/efficient_history_ft.json", "r", encoding="utf-8") as f:
        hft = json.load(f)

    plot_history_from_json(hfe, hft, title_prefix="EfficientNetB0",
                          save_path="results/figures/efficient_history.png")
    evaluate_confusion_report(model, val_gen, save_dir="results/figures", prefix="efficient")
    evaluate_with_threshold(model, val_gen, threshold_value=0.5, save_dir="results/figures", prefix="efficient")
