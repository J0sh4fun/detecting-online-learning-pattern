import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import csv

# Đường dẫn đến tệp dữ liệu của bạn
file_path = 'data/posture_dataset.csv'

def repair_and_process_dataset(path, threshold=0.005):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    # 1. Tải dữ liệu an toàn và sửa lỗi cấu trúc (Sửa lỗi ParserError)
    cleaned_rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Nếu header có 11 cột, loại bỏ 'hand_to_ear' (chỉ mục 5)
        if len(header) == 11:
            header.pop(5)
            
        for row in reader:
            if not row: continue 
            # Nếu hàng có 11 cột, loại bỏ phần tử thứ 6 để khớp với định dạng 10 cột mới
            if len(row) == 11:
                row.pop(5)
            
            if len(row) == len(header):
                cleaned_rows.append(row)

    # Chuyển đổi danh sách đã làm sạch thành DataFrame
    df = pd.DataFrame(cleaned_rows, columns=header)
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.dropna()
    
    original_load_count = len(df)
    print(f"Đã tải thành công {original_load_count} mẫu từ tệp.")

    # --- BƯỚC MỚI: Loại bỏ các nhãn không mong muốn ---
    labels_to_remove = ['Absence', 'Using Phone']
    df = df[~df['label'].isin(labels_to_remove)]
    after_filter_count = len(df)
    print(f"Đã loại bỏ {original_load_count - after_filter_count} mẫu thuộc nhãn Absence/Using Phone.")

    # 2. Loại bỏ các mẫu trùng lặp hoàn toàn
    df = df.drop_duplicates()
    
    # 3. Lọc các mẫu gần giống nhau (Dựa trên khoảng cách Euclidean)
    if not df.empty:
        features = df.drop('label', axis=1).values
        nn = NearestNeighbors(radius=threshold)
        nn.fit(features)
        
        adj_matrix = nn.radius_neighbors_graph(features).toarray()
        
        to_exclude = set()
        for i in range(len(adj_matrix)):
            if i not in to_exclude:
                neighbors = np.where(adj_matrix[i] == 1)[0]
                for n in neighbors:
                    if n != i:
                        to_exclude.add(n)
                        
        df_filtered = df.drop(df.index[list(to_exclude)])
    else:
        df_filtered = df

    # 4. Trộn ngẫu nhiên tập dữ liệu
    df_shuffled = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_count = len(df_shuffled)
    
    # 5. Ghi đè tệp gốc với dữ liệu sạch (10 cột)
    df_shuffled.to_csv(path, index=False)
    
    print(f"Xử lý hoàn tất.")
    print(f"Tổng số mẫu bị loại bỏ (nhãn cũ/trùng lặp/tương đồng): {original_load_count - final_count}")
    print(f"Kích thước tập dữ liệu cuối cùng: {final_count}")
    print(f"Danh sách các nhãn còn lại: {df_shuffled['label'].unique()}")
    print(f"Đã lưu tại: {path}")

# Thực thi
repair_and_process_dataset(file_path)