Dự án xây dựng hệ thống chẩn đoán bệnh gà từ ảnh bằng Deep Learning, sử dụng và so sánh 3 mô hình:
    ResNet50
    EfficientNetB0
    CNN
Project hỗ trợ:
    Huấn luyện mô hình
    Đánh giá & so sánh các chỉ số
    Dự đoán ảnh đơn
    Demo giao diện bằng Streamlit

- Yêu cầu hệ thống: Python 3.10/3.11
- Tạo môi trường ảo:
    + Bước 1: Mở terminal tại thư mục project 
    cd Nhom3
    + Bước 2: Tạo virtual environment
    python -m venv .venv
    + Bước 3: Kích hoạt môi trường
    .venv\Scripts\activate
- Cài đặt thư viện cần thiết: pip install -r requirements.txt
- Cách chạy từng thành phần:
    + Phân tích dữ liệu (EDA): 
    python -m src.eda
    + Train ResNet50: 
    python -m src.model_Resnet50_CaoThiNgocLuyen
    + Train EfficientNetB0:
    python -m src.model_EfficientNet_NguyenQuocBao
    + Train CNN:
    python -m src.model_CNN_LucDuyQuang
    + Đánh giá & so sánh 3 mô hình:
    python -m src.compare
    + Dự đoán ảnh đơn:
    python -m src.main
    + Chạy giao diện Streamlit:
    streamlit run app.py
- Kết quả:
    Model lưu trong models/
    History lưu trong results/history/
    Biểu đồ lưu trong results/figures/
