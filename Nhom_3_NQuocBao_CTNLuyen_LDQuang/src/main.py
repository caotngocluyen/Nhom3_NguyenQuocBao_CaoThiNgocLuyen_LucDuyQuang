import os # Thư viện tương tác với hệ điều hành
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image # Công cụ xử lý file ảnh

from .preprocessing import IMG_SIZE_RESNET_EFF, IMG_SIZE_CNN

# preprocess từng model
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficient
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet

# Danh sách nhãn phải khớp tuyệt đối với lúc train
CLASS_NAMES = ["Coccidiosis", "Healthy", "New Castle Disease", "Salmonella"]

def load_models():
    models_dict = {}

    # Efficient
    eff_path = "models/EfficienNetB0.keras"
    if os.path.exists(eff_path):
        models_dict["efficient"] = tf.keras.models.load_model(eff_path, compile=False)
    else:
        print(f"Không tìm thấy Efficient model: {eff_path}")

    # ResNet50
    res_path = "models/resnet50_final2.keras"
    if os.path.exists(res_path):
        models_dict["resnet50"] = tf.keras.models.load_model(res_path, compile=False)
    else:
        print(f"Không tìm thấy ResNet50 model: {res_path}")

    # CNN
    cnn_path = "models/CNN_final.keras"
    if os.path.exists(cnn_path):
        models_dict["cnn"] = tf.keras.models.load_model(cnn_path, compile=False)
    else:
        print(f"Không tìm thấy CNN model: {cnn_path}")

    return models_dict

def _predict_one(img_path, model, img_size, preprocess_fn=None, rescale_255=False):
    # 1. Load ảnh và resize
    img = image.load_img(img_path, target_size=img_size)

    # 2. Chuyển sang array + batch
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # 3. Preprocess
    if rescale_255:
        x = x / 255.0
    if preprocess_fn is not None:
        x = preprocess_fn(x)

    # 4. Predict
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds))
    conf = float(preds[0][idx]) * 100
    return idx, conf, preds

def predict_all(img_path):
    if not os.path.exists(img_path):
        print("Không thấy ảnh:", img_path)
        return

    models_dict = load_models()
    if len(models_dict) == 0:
        print("Không có model nào để predict.")
        return

    print("\n====================")
    print("KẾT QUẢ DỰ ĐOÁN")
    print("Ảnh:", os.path.basename(img_path))
    print("====================")

    # Efficient
    if "efficient" in models_dict:
        idx, conf, _ = _predict_one(
            img_path, models_dict["efficient"],
            IMG_SIZE_RESNET_EFF,
            preprocess_fn=preprocess_input_efficient
        )
        print(f"[EfficientNetB0] -> {CLASS_NAMES[idx]} ({conf:.2f}%)")

    # ResNet50
    if "resnet50" in models_dict:
        idx, conf, _ = _predict_one(
            img_path, models_dict["resnet50"],
            IMG_SIZE_RESNET_EFF,
            preprocess_fn=preprocess_input_resnet
        )
        print(f"[ResNet50]       -> {CLASS_NAMES[idx]} ({conf:.2f}%)")

    # CNN
    if "cnn" in models_dict:
        idx, conf, _ = _predict_one(
            img_path, models_dict["cnn"],
            IMG_SIZE_CNN,
            preprocess_fn=None,
            rescale_255=True
        )
        print(f"[CNN]           -> {CLASS_NAMES[idx]} ({conf:.2f}%)")

if __name__ == "__main__":
    # chạy: python -m src.main
    test_image = r"D:\Hocmay\ga\Train\salmo.289.jpg"
    predict_all(test_image)
