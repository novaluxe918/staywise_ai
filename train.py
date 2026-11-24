import pandas as pd

# Đọc dữ liệu từ file CSV
du_lieu = pd.read_csv("data.csv")

# Hiển thị toàn bộ dữ liệu
print("DANH SÁCH DỮ LIỆU:")
print(du_lieu)

# Hiển thị thông tin dữ liệu
print("\nTHÔNG TIN DỮ LIỆU:")
print(du_lieu.info())

# Kiểm tra dữ liệu bị thiếu
print("\nKIỂM TRA DỮ LIỆU THIẾU:")
print(du_lieu.isnull().sum())

# Thống kê dữ liệu
print("\nTHỐNG KÊ DỮ LIỆU:")
print(du_lieu.describe())