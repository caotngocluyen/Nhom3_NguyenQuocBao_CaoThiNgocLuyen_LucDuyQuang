import os
import json
import numpy as np # Thư viện xử lý mảng và số liệu
import matplotlib.pyplot as plt # Thư viện vẽ biểu đồ cơ bản
import seaborn as sns # Thư viện vẽ biểu đồ nhiệt (Heatmap)
from sklearn.metrics import confusion_matrix, classification_report # Các công cụ đo lường độ chính xác
import tensorflow as tf
from tensorflow.keras.preprocessing import image # Công cụ xử lý file ảnh

def _safe_load_history(path):
    if path is None: #Kiểm tra có đường truyền hay không
        return None
    if not os.path.exists(path): #Kiểm tra file có tồn tại không
        print(f"Không thấy history: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) #đọc nội dung JSON từ file object f

def plot_history_from_json(history_fe_json, history_ft_json=None,
                            title_prefix="Model", save_path=None):
    #Vẽ biểu đồ từ history JSON (để evaluation chạy độc lập, không cần train lại)
    #Kiểm tra có dữ liệu không
    if history_fe_json is None:
        print("Không có history_fe_json để vẽ.")
        return

    # Nếu có FT thì nối 
    if history_ft_json is not None:
        def merge_histories(h1, h2):
            out = {}
            for k in h1.keys(): #trả về danh sách key của dict (ví dụ: "accuracy", "loss", …)
                out[k] = list(h1.get(k, [])) + list(h2.get(k, [])) #lấy value theo key và ép về list
            return out

        hist = merge_histories(history_fe_json, history_ft_json)
        # Điểm đánh dấu kết thúc giai đoạn 1,tính số epoch FE để kẻ vạch đỏ
        len_fe = len(history_fe_json.get("accuracy", [])) #lấy danh sách accuracy của FE.
    else:
        hist = history_fe_json
        len_fe = None

    plt.figure(figsize=(14, 5))

    # Accuracy
    #tạo lưới subplot 1 hàng, 2 cột, chọn ô thứ 1
    plt.subplot(1, 2, 1)
    #vẽ train
    plt.plot(hist.get("accuracy", []), #Lấy danh sách accuracy theo epoch
             label="Train Accuracy", 
             marker='o', #Mỗi epoch được vẽ một chấm tròn
             markersize=3)
    #vẽ val
    plt.plot(hist.get("val_accuracy", []), label="Val Accuracy", marker='o', markersize=3)
    #vẽ vạch đỏ phân FE và FT
    if len_fe is not None and len_fe > 0: #
        plt.axvline(x=len_fe - 1, #vẽ đường thẳng đứng, -1 vì epoch bắt đầu vẽ từ 0
                     color='red', 
                     linestyle='--', 
                     label='Start Fine-tuning')
        
    plt.title(f"{title_prefix} - Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Loss
    #tạo lưới subplot 1 hàng, 2 cột, chọn ô thứ 2
    plt.subplot(1, 2, 2)
    plt.plot(hist.get("loss", []), label="Train Loss", marker='o', markersize=3)
    plt.plot(hist.get("val_loss", []), label="Val Loss", marker='o', markersize=3)
    if len_fe is not None and len_fe > 0:
        plt.axvline(x=len_fe - 1, color='red', linestyle='--', label='Start Fine-tuning')
    plt.title(f"{title_prefix} - Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout() #Tự động căn chỉnh khoảng cách, nếu ko có thì title đè lên suplot

    if save_path is not None: #nếu có truyền đường dẫn lưu file ảnh
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200) #Lưu hình vẽ hiện tại ra file
        print("Saved figure:", save_path)

    plt.show()

#tạo Confusion Matrix và vẽ heatmap
def evaluate_confusion_report(model, val_gen, save_dir="results/figures", prefix="model"):
    os.makedirs(save_dir, exist_ok=True)

    #dự đoán trên validation
    val_gen.reset() #đưa con trỏ về đầu
    pred_probs = model.predict(val_gen) #xác suất dự đoán cho mỗi lớp
    y_pred = np.argmax(pred_probs, axis=1) #Chọn lớp có xác suất lớn nhất cho từng ảnh
    y_true = val_gen.classes #Nhãn thật đã được generator gán sẵn

    # mapping index -> label
    #idx_to_label đảo ngược lại {0:'Coccidiosis'}
    idx_to_label = {v: k for k, v in val_gen.class_indices.items()} #Đảo dict để map index ==> label
    labels_sorted = [idx_to_label[i] for i in range(len(idx_to_label))] 
    #kq: labels_sorted = ["Coccidiosis", "Healthy", "Newcastle Disease", "Salmonella"]

    #Tính Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels_sorted,
        yticklabels=labels_sorted
    )

    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    #Lưu hình và hiển thị
    out_cm = os.path.join(save_dir, f"{prefix}_confusion_matrix.png") #Tạo đường dẫn file output
    plt.savefig(out_cm, dpi=200)
    plt.show()
    print("Saved:", out_cm)

    # Classification report
    print(classification_report(y_true, y_pred, target_names=labels_sorted))

def evaluate_with_threshold(model, val_gen, threshold_value=0.5, s_idx=3, save_dir="results/figures", prefix="efficient"):
    """Hàm dự đoán với kỹ thuật Threshold Tuning (giữ theo code Efficient của bạn)"""
    os.makedirs(save_dir, exist_ok=True)

    val_gen.reset() # Đặt lại bộ nạp ảnh về vị trí đầu tiên
    y_pred_probs = model.predict(val_gen, verbose=1) # Dự đoán xác suất cho toàn bộ tập Validation

    y_pred_tuned = []
    # Duyệt qua từng kết quả dự đoán (dạng xác suất)
    for prob in y_pred_probs:
        # Nếu xác suất Salmonella không vượt qua ngưỡng tin cậy mới
        if prob[s_idx] < threshold_value:
            p_temp = prob.copy()
            p_temp[s_idx] = 0 # Ép lớp Salmonella về 0
            y_pred_tuned.append(np.argmax(p_temp)) # Chọn lớp có xác suất cao thứ nhì
        else:
            y_pred_tuned.append(np.argmax(prob)) # Nếu vượt ngưỡng thì tin tưởng kết quả mô hình

    y_pred_tuned = np.array(y_pred_tuned)
    y_true = val_gen.classes # Nhãn thực tế từ tập dữ liệu

    # Lấy danh sách tên lớp từ generator
    class_names = list(val_gen.class_indices.keys())

    # Tính toán và vẽ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_tuned)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Salmonella Threshold: {threshold_value})', fontsize=16)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    out_cm = os.path.join(save_dir, f"{prefix}_threshold_{threshold_value}_cm.png")
    plt.savefig(out_cm, dpi=200)
    plt.show()
    print("Saved:", out_cm)

    # In báo cáo
    print(f"\nClassification Report (Threshold = {threshold_value}):")
    report = classification_report(y_true, y_pred_tuned, target_names=class_names)
    print(report)
    return report

def predict_image_resnet_or_eff(img_path, model, preprocess_fn, img_size, class_indices):
    # mapping index -> label như trên
    idx_to_label = {v: k for k, v in class_indices.items()}

    # 1. Load ảnh và resize
    img = image.load_img(img_path, target_size=img_size)

    # 2. Chuyển sang array + batch
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # 3. Preprocess chuẩn
    x = preprocess_fn(x)

    # 4. Predict
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds))
    predicted_class = idx_to_label[idx]
    confidence = preds[0][idx] * 100

    # 5. Hiển thị ảnh + kết quả
    plt.imshow(img)
    plt.title(f"Dự đoán: {predicted_class} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()

    print(f"Ảnh được dự đoán là: {predicted_class}")
    print(f"Độ tin cậy: {confidence:.2f}%")

    return predicted_class, confidence, preds

#chạy độc lập, chỉ load model
def run_evaluation(model_name="resnet50"):
    #Tạo thư mục lưu hình
    save_dir = "results/figures"
    os.makedirs(save_dir, exist_ok=True)

    if model_name == "resnet50":
        #Khai báo đường dẫn file model và history
        model_path = "models/resnet50_final2.keras"
        hist_fe_path = "results/history/resnet50_history_fe.json"
        hist_ft_path = "results/history/resnet50_history_ft.json"

        from .preprocessing import load_and_split_data
        from .feature import get_val_generator_resnet50

        #Tách dữ liệu train/val
        train_df, val_df = load_and_split_data()
        #Tạo generator validation cho ResNet50
        val_gen = get_val_generator_resnet50(val_df)

        #load model
        loaded = tf.keras.models.load_model(model_path, compile=False)

        # Vẽ history từ JSON
        hfe = _safe_load_history(hist_fe_path)
        hft = _safe_load_history(hist_ft_path)
        plot_history_from_json(hfe, hft, title_prefix="ResNet50",
                              save_path=os.path.join(save_dir, "resnet50_history.png"))

        # Confusion matrix + report
        evaluate_confusion_report(loaded, val_gen, save_dir=save_dir, prefix="resnet50")

    elif model_name == "efficient":
        model_path = "models/EfficienNetB0.keras"
        hist_fe_path = "results/history/efficient_history_fe.json"
        hist_ft_path = "results/history/efficient_history_ft.json"

        from .preprocessing import load_and_split_data
        from .feature import get_val_generator_efficient

        train_df, val_df = load_and_split_data()
        val_gen = get_val_generator_efficient(val_df)

        loaded = tf.keras.models.load_model(model_path, compile=False)

        hfe = _safe_load_history(hist_fe_path)
        hft = _safe_load_history(hist_ft_path)
        plot_history_from_json(hfe, hft, title_prefix="EfficientNetB0",
                              save_path=os.path.join(save_dir, "efficient_history.png"))

        evaluate_confusion_report(loaded, val_gen, save_dir=save_dir, prefix="efficient")

        # Threshold tuning
        evaluate_with_threshold(loaded, val_gen, threshold_value=0.5, save_dir=save_dir, prefix="efficient")

    elif model_name == "cnn":
        model_path = "models/CNN_final.keras"
        hist_path = "results/history/cnn_history.json"

        from .preprocessing import load_and_split_data
        from .feature import get_train_val_generators_cnn

        train_df, val_df = load_and_split_data()
        # CNN split ngay trong generator train_datagen
        train_gen, val_gen = get_train_val_generators_cnn(train_df, val_split=0.2)

        loaded = tf.keras.models.load_model(model_path, compile=False)

        h = _safe_load_history(hist_path)
        plot_history_from_json(h, None, title_prefix="CNN",
                              save_path=os.path.join(save_dir, "cnn_history.png"))

        evaluate_confusion_report(loaded, val_gen, save_dir=save_dir, prefix="cnn")

    else:
        print("model_name phải là: resnet50 | efficient | cnn")

if __name__ == "__main__":
    # chạy: python -m src.evaluation
    # muốn đổi model: sửa model_name ở đây
    run_evaluation(model_name="cnn")
