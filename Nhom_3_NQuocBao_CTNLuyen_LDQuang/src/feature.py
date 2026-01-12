from tensorflow.keras.preprocessing.image import ImageDataGenerator #đọc ảnh theo batch + augmentation

# Import config chung
from .preprocessing import (
    IMG_SIZE_RESNET_EFF, IMG_SIZE_CNN, BATCH_SIZE, SEED
)

# ===== EfficientNet preprocess =====
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficient

# ===== ResNet50 preprocess =====
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet

# =============================
# EFFICIENTNET
# =============================
def get_train_generator_efficient(train_df):
    #Tạo generator cho tập train với kỹ thuật Augmentation mạnh
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_efficient, # Chuẩn hóa màu sắc ảnh về dạng mô hình hiểu được
        rotation_range=30, # Xoay ảnh ngẫu nhiên trong khoảng 30 độ
        width_shift_range=0.2, # Dịch chuyển ảnh sang trái/phải 20% chiều rộng
        height_shift_range=0.2, # Dịch chuyển ảnh lên/xuống 20% chiều cao
        shear_range=0.15, # Làm nghiêng ảnh
        zoom_range=0.2, # Phóng to hoặc thu nhỏ ảnh 20%
        horizontal_flip=True, # Lật ngược ảnh theo chiều ngang
        brightness_range=[0.7, 1.3], # Giải quyết lỗi ánh sáng
        fill_mode='nearest' # Xử lý các vùng trống sau khi xoay/dịch ảnh
    )

    # Trả về bộ nạp ảnh đã được cấu hình
    return train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepath", # Cột chứa đường dẫn ảnh trong DataFrame
        y_col="label", # Cột chứa nhãn bệnh
        target_size=IMG_SIZE_RESNET_EFF, # Đưa ảnh về kích thước chuẩn 224x224
        batch_size=BATCH_SIZE, # Nạp ảnh theo từng lô (ví dụ 32 tấm/lần)
        class_mode="categorical", # Phân loại đa lớp (đầu ra là vector xác suất)
        shuffle=True, # Xáo trộn thứ tự ảnh sau mỗi vòng lặp để mô hình không overfitting
        seed=SEED
    )

def get_val_generator_efficient(val_df):
    #Tạo generator cho tập validation
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_efficient)
    return val_datagen.flow_from_dataframe(
        val_df,
        x_col="filepath", # Cột chứa đường dẫn ảnh
        y_col="label",    # Cột chứa nhãn bệnh
        target_size=IMG_SIZE_RESNET_EFF, # Đưa ảnh về kích thước 224x224
        batch_size=BATCH_SIZE, # Gom ảnh thành các lô 32 tấm
        class_mode="categorical", # Mã hóa nhãn dạng One-hot (đa lớp)
        shuffle=False # Không xáo trộn để vẽ Confusion Matrix
    )

# =============================
# RESNET50
# =============================
def get_train_generator_resnet50(train_df):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_resnet, #chuẩn hoá ảnh theo đúng kiểu ResNet50 pretrained
        rotation_range=15, #xoay +-15 độ
        width_shift_range=0.1, # Dịch ngang lên đến 10% chiều rộng ảnh --> học được vị trí vật thể không cố định trong ảnh
        height_shift_range=0.1, # Dịch dọc lên đến 10% chiều cao ảnh --> tăng khả năng nhận dạng bất chấp vật thể ở vị trí khác
        zoom_range=0.1, # Phóng to/thu nhỏ ảnh trong khoảng ±20%
        horizontal_flip=True #Lật ngang --> Hữu ích cho ảnh đối xứng
    )

    # Tạo generator cho tập huấn luyện
    return train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE_RESNET_EFF,
        class_mode="categorical", #nhãn được one-hot (vd [0,0,1,0])
        batch_size=BATCH_SIZE,
        shuffle=True, #Trộn thứ tự ảnh mỗi epoch, để model không học theo thứ tự cố định
        seed=SEED
    )

def get_val_generator_resnet50(val_df):
    #validation KHÔNG dùng augmentation do phải dùng dữ liệu giống thật ngoài đời
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_resnet
    )

    # Tạo generator cho tập validation
    return val_datagen.flow_from_dataframe(
        val_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE_RESNET_EFF,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False #Giữ nguyên thứ tự ảnh trong validation
    )

# =============================
# CNN
# =============================
def get_train_val_generators_cnn(train_df, val_split=0.2):
    # 3. Tiền xử lý & Tăng cường dữ liệu (Augmentation)
    # ImageDataGenerator trong Keras dùng để chuẩn bị và biến đổi dữ liệu ảnh đầu vào trước khi đưa vào mô hình
    # Chuẩn hóa dữ liệu (Normalization) → giúp mô hình học nhanh và ổn định hơn.
    # Tăng cường dữ liệu (Data Augmentation) → tạo thêm biến thể mới của ảnh để giúp mô hình tổng quát hóa tốt hơn, tránh overfitting.
    train_datagen = ImageDataGenerator(
        rescale=1./255,            # Chuẩn hóa giá trị pixel về [0, 1]
        rotation_range=20,         # Xoay nhẹ ảnh trong khoảng ±20 độ --> Giúp mô hình chống lại sự lệch góc, học vật thể vẫn giống nhau dù xoay nhẹ
        width_shift_range=0.2,     # Dịch ngang lên đến 20% chiều rộng ảnh --> học được vị trí vật thể không cố định trong ảnh
        height_shift_range=0.2,    # Dịch dọc lên đến 20% chiều cao ảnh --> tăng khả năng nhận dạng bất chấp vật thể ở vị trí khác
        zoom_range=0.2,            # Phóng to/thu nhỏ ảnh trong khoảng ±20%
        horizontal_flip=True,      # Lật ngang --> Hữu ích cho ảnh đối xứng
        validation_split=val_split # Tách tập validation
    )

    # Tạo generator cho tập huấn luyện
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE_CNN,
        batch_size=BATCH_SIZE,
        subset='training',
        class_mode='categorical'
    )

    # Tạo generator cho tập validation
    val_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE_CNN,
        batch_size=BATCH_SIZE,
        subset='validation',
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen

def get_val_generator_cnn(val_df):
    # Tạo generator cho tập validation của CNN (KHÔNG dùng validation_split nữa),
    # vì compare cần dùng đúng val_df đã tách sẵn để công bằng với ResNet/Eff.
    val_datagen = ImageDataGenerator(
        rescale=1./255  # Chuẩn hóa pixel về [0,1]
    )

    return val_datagen.flow_from_dataframe(
        val_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE_CNN,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
