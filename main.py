
from fastapi import FastAPI
from schemas import DuDoanGiaRequest, DuDoanGiaResponse

import joblib
import pandas as pd

app = FastAPI()

# 1. Đọc mô hình đã huấn luyện
mo_hinh = joblib.load("mo_hinh_gia_phong.pkl")


@app.get("/")
def trang_chu():
    return {
        "message": "StayWise AI Service dang hoat dong"
    }


@app.post("/predict", response_model=DuDoanGiaResponse)
def du_doan_gia(data: DuDoanGiaRequest):

    # 2. Chuyển dữ liệu nhận được thành DataFrame
    du_lieu = pd.DataFrame([{
        "dienTich": data.dienTich,
        "soNguoi": data.soNguoi,
        "loaiPhong": data.loaiPhong,
        "tinhThanh": data.tinhThanh,
        "thang": data.ngayApDung.month
    }])

    # 3. Sử dụng mô hình để dự đoán
    gia_du_doan = mo_hinh.predict(du_lieu)[0]

    # 4. Trả kết quả
    return {
        "maPhong": data.maPhong,
        "ngayApDung": data.ngayApDung,
        "giaDuDoan": round(float(gia_du_doan), 0),
        "doTinCay": 0,
        "phienBan": "random_forest_v1"
    }
