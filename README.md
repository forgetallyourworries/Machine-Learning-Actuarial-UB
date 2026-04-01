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

# Hasil Ujicoba atau Test terhadap BBCA Stock Movement Prediction using Support Vector Machine (SVM)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange)
![yfinance](https://img.shields.io/badge/yfinance-Financial_Data-green)

## 📌 Tentang Proyek
Proyek ini mengimplementasikan algoritma **Support Vector Machine (SVM)** untuk memprediksi arah pergerakan harian harga saham PT Bank Central Asia Tbk (BBCA). Alih-alih memprediksi harga pasti, model ini melakukan klasifikasi biner: apakah harga penutupan esok hari akan **Naik (1)** atau **Turun/Tetap (0)**.

Proyek ini juga membandingkan tiga konfigurasi kernel SVM (Linear, RBF, dan Polynomial) untuk menguji hipotesis matematis mana yang paling cocok dengan perilaku fluktuasi pasar saham.

## 🧠 Filosofi di Balik Program (Apa & Mengapa?)

### Sebenarnya Apa Sih Program Ini?
Program ini bertindak layaknya seorang analis kuantitatif yang tidak pernah tidur. Sistem kecerdasan buatan (*machine learning*) ini bertugas menjawab satu pertanyaan spesifik berdasarkan data historis: **"Melihat tren hari ini dan beberapa hari ke belakang, ke mana arah saham BBCA besok?"**
Program tidak menganalisis berita ekonomi atau laporan keuangan. Model ini murni mengandalkan indikator statistik historis (seperti *Moving Average* dan tingkat volatilitas) untuk mencari pola matematis yang tersembunyi di balik pergerakan harga.

### Mengapa Menggunakan Support Vector Machine (SVM)?
Pasar saham itu sangat bising (*noisy*) dan penuh anomali. SVM dipilih karena algoritma ini dirancang khusus untuk mencari **batas keputusan (*hyperplane*)** terbaik yang memisahkan dua kelas data (hari "Naik" dan hari "Turun"). 
Dari sudut pandang matematika, proses ini pada dasarnya adalah problem optimasi numerik tingkat tinggi (memiliki kedekatan filosofis dengan metode optimasi seperti DFP atau BFGS) di mana algoritma berusaha mencari titik ekstremum untuk meminimalkan *error* klasifikasi sekaligus memaksimalkan margin toleransi antar data.

## 🔄 Pipeline Machine Learning

Alur kerja data dari pengambilan *real-time* hingga prediksi direpresentasikan dalam diagram berikut:

```mermaid
graph TD
    A([1. Data Extraction]) -->|yfinance API| B[Data Historis BBCA]
    
    B --> C[2. Feature Engineering]
    C -->|Hitung| C1(Return Harian)
    C -->|Hitung| C2(SMA 10 & 30 Hari)
    C -->|Hitung| C3(Volatilitas 10 Hari)
    C -->|Shift Data| C4(Label: Target Besok)
    
    C1 & C2 & C3 & C4 --> D[3. Train-Test Split]
    D -.->|Wajib: shuffle=False| D1(Time-Series Split)
    
    D1 --> E[4. Scikit-Learn Pipeline]
    E -->|StandardScaler| F(Normalisasi Skala Data)
    F -->|Algoritma| G{Model SVM}
    
    G -->|Versi 1| H1[Kernel Linear]
    G -->|Versi 2| H2[Kernel RBF]
    G -->|Versi 3| H3[Kernel Polynomial]
    
    H1 & H2 & H3 --> I[5. Evaluasi]
    I -->|Output| J[Akurasi & Classification Report]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#ff9,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
```

## 🧪 Skenario Eksperimen & Hasil (Periode 2022-2026)

Kami menguji tiga versi arsitektur model:

1. **Versi 1: Linear Kernel (C=0.1)**
   * **Konsep:** Margin lebar untuk menoleransi *noise* pasar saham.
   * **Hasil (58.42%):** Akurasi tertinggi. Menunjukkan bahwa model sederhana yang tidak mudah *overfitting* bekerja paling baik di pasar yang sangat fluktuatif.
2. **Versi 2: RBF Kernel (C=10.0, Gamma=0.1)**
   * **Konsep:** Model kompleks yang dipaksa menghafal pola dengan ketat.
   * **Hasil (58.06%):** Terkena *overfitting*. Pola masa lalu yang dihafal model tidak selalu terulang secara identik di masa depan.
3. **Versi 3: Polynomial Kernel (Degree 3)**
   * **Konsep:** Mencoba mencocokkan pola pergerakan harga dengan kurva polinomial derajat tiga.
   * **Hasil (55.91%):** Akurasi terendah, membuktikan bahwa pergerakan pasar saham tidak mengikuti fungsi matematis yang kaku.

## 💡 Kesimpulan Pembelajaran
Akurasi tertinggi berada di angka ~58%. Dalam ranah *quant trading*, mampu memprediksi arah pasar lebih baik dari sekadar peluang acak (50%) adalah sebuah pencapaian. Namun, hal ini juga membuktikan secara kuantitatif bahwa *timing the market* secara harian sangat sulit. Model ini secara tidak langsung memvalidasi bahwa strategi investasi rutin berkala (*Dollar Cost Averaging*) secara konsisten merupakan pendekatan yang lebih stabil menghadapi volatilitas pasar.

## 🚀 Cara Penggunaan
1. Pastikan Anda telah menginstal pustaka yang dibutuhkan:
   `pip install yfinance pandas numpy scikit-learn`
2. Jalankan skrip Python utama. Program akan otomatis mengunduh data terbaru dari Yahoo Finance dan melatih model secara *real-time*.
