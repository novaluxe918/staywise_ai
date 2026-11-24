from pydantic import BaseModel
from datetime import date


class DuDoanGiaRequest(BaseModel):
    maPhong: int
    ngayApDung: date
    giaHienTai: float
    dienTich: float
    soNguoi: int
    loaiPhong: str
    tinhThanh: str
    
    
class DuDoanGiaResponse(BaseModel):
    maPhong: int
    ngayApDung: date
    giaDuDoan: float
    doTinCay: float
    phienBan: str