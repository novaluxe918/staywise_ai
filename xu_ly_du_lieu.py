import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# 1. Đọc dữ liệu
du_lieu = pd.read_csv("data.csv")

# 2. Tách dữ liệu đầu vào X
X = du_lieu[
    ["dienTich", "soNguoi", "loaiPhong", "tinhThanh", "thang"]
]

# 3. Tách kết quả cần dự đoán y
y = du_lieu["giaPhong"]

# 4. Xác định các cột chứa dữ liệu chữ
cot_chu = ["loaiPhong", "tinhThanh"]

# 5. Chuyển dữ liệu chữ thành dữ liệu số
bo_chuyen_doi = ColumnTransformer(
    transformers=[
        ("ma_hoa", OneHotEncoder(handle_unknown="ignore"), cot_chu)
    ],
    remainder="passthrough"
)

# 6. Thực hiện chuyển đổi
X_da_xu_ly = bo_chuyen_doi.fit_transform(X)

# 7. Hiển thị kết quả
print("DỮ LIỆU ĐẦU VÀO BAN ĐẦU:")
print(X)

print("\nDỮ LIỆU SAU KHI CHUYỂN ĐỔI:")
print(X_da_xu_ly)

print("\nTÊN CÁC ĐẶC TRƯNG SAU CHUYỂN ĐỔI:")
print(bo_chuyen_doi.get_feature_names_out())

print("\nGIÁ PHÒNG CẦN DỰ ĐOÁN:")
print(y)