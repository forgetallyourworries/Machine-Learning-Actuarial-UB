# Klasifikasi Bunga Iris menggunakan Support Vector Machine (SVM)

Proyek sederhana ini mendemonstrasikan *pipeline machine learning* dasar untuk melakukan klasifikasi pada dataset bunga Iris menggunakan algoritma **Support Vector Classifier (SVC)** dari *library* `scikit-learn`.

## 🔄 Pipeline Model

Berikut adalah alur data dari tahap pemuatan hingga evaluasi:

```mermaid
graph TD
    A([0. Import Libraries]) --> B[1. Load Dataset Iris]
    
    B --> |X = Fitur, y = Label| C[2. Train-Test Split]
    
    C --> |70% Data Latih| C1[(X_train, y_train)]
    C --> |30% Data Uji| C2[(X_test, y_test)]
    
    C1 --> D[3. Train SVM Model]
    D --> |SVC kernel='rbf'| D1{Model Terlatih}
    
    C2 --> |Input X_test| E[4. Prediksi]
    D1 --> E
    
    E --> |Menghasilkan y_pred| F[5. Evaluasi Model]
    C2 -.-> |Bandingkan dengan y_test| F
    
    F --> F1[Accuracy Score]
    F --> F2[Confusion Matrix]
    F --> F3[Classification Report]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style C1 fill:#dfd,stroke:#333,stroke-width:2px
    style C2 fill:#fdd,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style D1 fill:#ff9,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px
```

## 💻 Kode Program

Pastikan sudah menginstal `scikit-learn` sebelum menjalankan program ini. instalnya dengan perintah `pip install scikit-learn`.

```python
# Import libraries
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset Iris
data = load_iris()
X = data.data # Fitur (sepal length, sepal width, petal length, petal width)
y = data.target # Label (setosa, versicolor, virginica)

# 2. Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train SVM model dengan kernel RBF
model = SVC(kernel='rbf', C=1.0) # Kernel RBF dan C=1.0
model.fit(X_train, y_train)

# 4. Prediksi hasil pada data uji
y_pred = model.predict(X_test)

# 5. Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output hasil evaluasi
print(f'Accuracy: {accuracy * 100:.2f}%')
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
```

## 📊 Hasil Evaluasi

Saat program dijalankan, model akan mengeluarkan metrik evaluasi berupa tingkat akurasi (dalam persen), *confusion matrix* untuk melihat detail tebakan per kelas, dan *classification report* yang memuat *precision, recall,* dan *f1-score*.
