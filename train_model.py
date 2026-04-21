# Tệp: train_model.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Tạo thư mục chứa mô hình nếu chưa có
os.makedirs("models", exist_ok=True)

def main():
    print("=== BƯỚC 1: TẢI & TIỀN XỬ LÝ DỮ LIỆU ===")
    try:
        df = pd.read_csv('data/posture_dataset.csv')
    except FileNotFoundError:
        df = pd.read_csv('posture_dataset.csv') # Dự phòng nếu file để ở thư mục ngoài cùng
        
    df = df.dropna()
    
    # Tách Đặc trưng (X) và Nhãn (y)
    X = df.drop('label', axis=1)
    y = df['label']
    
    print(f"Tổng số mẫu hợp lệ: {len(df)}")
    print("Phân bố nhãn:")
    print(y.value_counts())

    # Chia tập Huấn luyện (80%) và tập Kiểm thử (20%)
    # stratify=y giúp đảm bảo tỷ lệ các lớp trong tập test giống hệt tập train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n=== BƯỚC 2: CHUẨN HÓA ĐẶC TRƯNG (SCALING) ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Lưu bộ scaler để Tuần 4 sử dụng
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Đã lưu 'scaler.pkl'")

    print("\n=== BƯỚC 3: HUẤN LUYỆN MÔ HÌNH ===")
    # 1. Random Forest (Sử dụng class_weight='balanced' để bù đắp cho lớp ít dữ liệu)
    print("Đang huấn luyện Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=15, class_weight='balanced', random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)

    # 2. Support Vector Machine (SVM)
    print("Đang huấn luyện SVM...")
    svm_model = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)

    print("\n=== BƯỚC 4: ĐÁNH GIÁ KẾT QUẢ ===")
    print(f"Độ chính xác của Random Forest: {rf_acc * 100:.2f}%")
    print(f"Độ chính xác của SVM:           {svm_acc * 100:.2f}%")

    # Chọn mô hình tốt nhất
    if rf_acc >= svm_acc:
        best_model = rf_model
        best_pred = rf_pred
        model_name = "Random Forest"
    else:
        best_model = svm_model
        best_pred = svm_pred
        model_name = "SVM"

    # Lưu mô hình tốt nhất
    joblib.dump(best_model, 'models/best_posture_model.pkl')
    print(f"\n=> Đã lưu mô hình tốt nhất ({model_name}) vào 'models/best_posture_model.pkl'")

    print("\n=== BÁO CÁO CHI TIẾT (CLASSIFICATION REPORT) ===")
    print(classification_report(y_test, best_pred))

if __name__ == "__main__":
    main()
