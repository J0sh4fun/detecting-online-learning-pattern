import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
from scipy import stats

# Set visualization style
sns.set(style="whitegrid")

# Loading the dataset provided in your project
try:
    df = pd.read_csv('/kaggle/input/datasets/minorin2847/posture-dataset/posture_dataset.csv')
except:
    # Fallback for local testing
    df = pd.read_csv('data/posture_dataset.csv')

df = df.dropna()
print(f"Total valid samples: {len(df)}")
df.head()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='label', palette='viridis')
plt.title('Distribution of Posture Labels')
plt.xlabel('Posture Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12, 10))
# Calculate correlation on numeric columns only
corr = df.drop('label', axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# Separate Features (X) and Labels (y)
X = df.drop('label', axis=1)
y = df['label']

# Split Training (80%) and Testing (20%)
# Stratify ensures the class distribution is preserved in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for inference use
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

# 1. Random Forest Training
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# 2. Support Vector Machine (SVM) Training
print("Training SVM...")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    class_weight='balanced',
    probability=True,
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)
svm_pred = svm_model.predict(X_test_scaled)

rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)

print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
print(f"SVM Accuracy: {svm_acc * 100:.2f}%")

# 1. R^2 Evaluation (Caution: R^2 is for Regression)
# To calculate R^2 for classification, we treat the categorical labels as integers.
# Note: This is not a standard way to evaluate a classifier.
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)
rf_pred_encoded = le.transform(rf_model.predict(X_test_scaled))

rf_r2 = r2_score(y_test_encoded, rf_pred_encoded)
print(f"Random Forest Pseudo-R^2 (based on encoded labels): {rf_r2:.4f}")

# 2. Statistical Significance (p-value) Comparison
# We use 10-fold cross-validation to get a distribution of accuracy scores
# and then perform a Paired T-Test to see if the models are significantly different.
print("\nCalculating p-value for model comparison (10-fold CV)...")
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=kfold)
svm_cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=kfold)

# Perform Paired T-Test
t_stat, p_value = stats.ttest_rel(rf_cv_scores, svm_cv_scores)

print(f"Random Forest CV Mean Accuracy: {rf_cv_scores.mean():.4f}")
print(f"SVM CV Mean Accuracy:           {svm_cv_scores.mean():.4f}")
print(f"Comparison p-value:             {p_value:.2e}")

if p_value < 0.05:
    print("=> Result: The performance difference is statistically significant (p < 0.05).")
else:
    print("=> Result: The performance difference is NOT statistically significant (p >= 0.05).")

# Select best model
best_model = rf_model if rf_acc >= svm_acc else svm_model
joblib.dump(best_model, 'models/best_posture_model.pkl')

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, rf_model.predict(X_test_scaled))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf_model.classes_,
            yticklabels=rf_model.classes_)
plt.title('Confusion Matrix: Random Forest')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12, 6))
plt.title('Feature Importance for Posture Detection')
plt.bar(range(X.shape[1]), importances[indices], align='center', color='teal')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
