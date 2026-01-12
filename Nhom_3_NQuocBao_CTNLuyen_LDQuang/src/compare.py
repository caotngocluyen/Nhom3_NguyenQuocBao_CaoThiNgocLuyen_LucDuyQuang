# src/compare.py
import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, f1_score

# Import lại preprocessing + generator
from .preprocessing import load_and_split_data
from .feature import (
    get_val_generator_resnet50,
    get_val_generator_efficient,
    get_val_generator_cnn
)

# =========================================================
# FIX load model có dùng Focal Loss (EfficientNet)
# =========================================================
def focal_loss_fixed(y_true, y_pred):
    return y_pred

def load_model_safe(path):
    return tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={"focal_loss_fixed": focal_loss_fixed}
    )

# =========================================================
# ĐÁNH GIÁ 1 MODEL – NHÓM CHỈ SỐ TỔNG QUAN
# =========================================================
def evaluate_summary(model, val_gen):
    val_gen.reset()

    probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = val_gen.classes

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "MacroF1": f1_score(y_true, y_pred, average="macro"),
        "WeightedF1": f1_score(y_true, y_pred, average="weighted"),
        "N": len(y_true)
    }

# =========================================================
# SO SÁNH 3 MÔ HÌNH
# =========================================================
def compare_models():
    print("\n Load dữ liệu validation chung cho 3 mô hình...")

    # Dùng cùng val_df để công bằng
    train_df, val_df = load_and_split_data()

    # Generator đúng cho từng model
    val_res = get_val_generator_resnet50(val_df)
    val_eff = get_val_generator_efficient(val_df)
    val_cnn = get_val_generator_cnn(val_df)

    # Đường dẫn model (SỬA ĐÚNG TÊN FILE BẠN CÓ)
    model_paths = {
        "ResNet50": os.path.join("models", "resnet50_final2.keras"),
        "EfficientNetB0": os.path.join("models", "EfficienNetB0.keras"),
        "CNN": os.path.join("models", "CNN_final.keras"),
    }

    results = []

    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"Không tìm thấy model: {path} → bỏ qua {name}")
            continue

        model = load_model_safe(path)

        if name == "ResNet50":
            r = evaluate_summary(model, val_res)
        elif name == "EfficientNetB0":
            r = evaluate_summary(model, val_eff)
        else:
            r = evaluate_summary(model, val_cnn)

        r["Model"] = name
        results.append(r)

    # =====================================================
    # IN BẢNG SO SÁNH CUỐI
    # =====================================================
    if not results:
        print("\nKhông có model nào được đánh giá.")
        return

    print("\n================= SO SÁNH TỔNG QUAN =================")
    print(f"{'Model':18} | Acc   | MacroF1 | WeightF1")
    print("-" * 55)

    # Sắp xếp theo Macro F1 (quan trọng nhất)
    results = sorted(results, key=lambda x: x["MacroF1"], reverse=True)

    for r in results:
        print(
            f"{r['Model']:18} | "
            f"{r['Accuracy']:.4f} | "
            f"{r['MacroF1']:.4f} | "
            f"{r['WeightedF1']:.4f}"
        )

if __name__ == "__main__":
    # chạy: python -m src.compare
    compare_models()
