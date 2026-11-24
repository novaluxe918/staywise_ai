
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

# 5. Chuyển dữ liệu chữ thành số
bo_chuyen_doi = ColumnTransformer(
    transformers=[
        ("ma_hoa", OneHotEncoder(handle_unknown="ignore"), cot_chu)
    ],
    remainder="passthrough"
)

# 6. Tạo mô hình Random Forest
mo_hinh = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# 7. Kết hợp xử lý dữ liệu và mô hình
quy_trinh = Pipeline([
    ("xu_ly", bo_chuyen_doi),
    ("mo_hinh", mo_hinh)
])

# 8. Chia dữ liệu thành 4 phần để kiểm tra chéo
kfold = KFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

# 9. Dự đoán bằng phương pháp kiểm tra chéo
gia_du_doan = cross_val_predict(
    quy_trinh,
    X,
    y,
    cv=kfold
)

# 10. Tính các chỉ số đánh giá
mae = mean_absolute_error(y, gia_du_doan)

rmse = mean_squared_error(
    y,
    gia_du_doan
) ** 0.5

r2 = r2_score(y, gia_du_doan)

# 11. Hiển thị kết quả
print("===== KET QUA DANH GIA MO HINH =====")

print(f"MAE: {mae:,.0f} VNĐ")
print(f"RMSE: {rmse:,.0f} VNĐ")
print(f"R2: {r2:.4f}")

# 12. So sánh giá thực tế và giá dự đoán
bang_so_sanh = pd.DataFrame({
    "Gia thuc te": y,
    "Gia du doan": gia_du_doan,
    "Sai lech": abs(y - gia_du_doan)
})

print("\n===== BANG SO SANH =====")
print(bang_so_sanh)

# 13. Huấn luyện lại bằng toàn bộ dữ liệu
quy_trinh.fit(X, y)

# 14. Lưu mô hình
joblib.dump(quy_trinh, "mo_hinh_gia_phong.pkl")

print("\nDA DANH GIA VA LUU MO HINH THANH CONG!")
