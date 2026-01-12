import os
import matplotlib.pyplot as plt # vẽ biểu đồ loss/accuracy và hiển thị ảnh
from PIL import Image #mở ảnh từ file

from .preprocessing import load_dataframe

def run_eda(save_dir="results/figures"): 
    os.makedirs(save_dir, exist_ok=True) #tạo folder nếu chưa tồn tại, không lỗi nếu folder đã có sẵn.

    df = load_dataframe()

    #So sánh phân bố lớp
    df["label"].value_counts().plot(kind="bar") #đếm số lượng ảnh của mỗi lớp.
    plt.title("Phân bố số lượng ảnh theo lớp")
    plt.xlabel("Lớp")
    plt.ylabel("Số ảnh")
    plt.tight_layout() #tự canh bố cục để chữ không bị cắt.
    out1 = os.path.join(save_dir, "eda_class_distribution.png") #ghép đường dẫn
    plt.savefig(out1, dpi=200) #lưu biểu đồ
    plt.show()

    #Hiển thị 1 vài ảnh mẫu
    fig = plt.figure(figsize=(8,8))
    labels = df["label"].unique()

    for i, label in enumerate(labels):
        sample = df[df["label"] == label].sample(1).iloc[0] #lọc ra các dòng thuộc lớp label, 
        #chọn ngẫu nhiên 1 dòng trong lớp đó, rồi lấy dòng đầu
        ax = fig.add_subplot(2,2,i+1) #Tạo đúng lưới 2x2 cho 4 lớp.
        ax.imshow(Image.open(sample["filepath"])) #Vẽ ảnh đó lên subplot ax
        ax.set_title(label)
        ax.axis("off") #Tắt trục tọa độ x,y, số

    plt.tight_layout() #Tự động căn chỉnh bố cục
    out2 = os.path.join(save_dir, "eda_sample_images.png")
    plt.savefig(out2, dpi=200)
    plt.show()

    print("Saved:", out1)
    print("Saved:", out2)

if __name__ == "__main__":
    # chạy: python -m src.eda
    run_eda()
