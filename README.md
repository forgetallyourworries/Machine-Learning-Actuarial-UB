# Modul Responsi Pengantar Pembelajaran Mesin (Machine Learning) 🤖

Repositori ini berisi panduan, materi, dan kode implementasi untuk mata kuliah **Responsi Pengantar Pembelajaran Mesin** [1]. 

**Penyusun:** Hilmi Aziz Bukhori S.kom., M.T.  
**Program Studi:** Sarjana Ilmu Aktuaria, Departemen Matematika  
**Institusi:** Fakultas Matematika dan Ilmu Pengetahuan Alam, Universitas Brawijaya (2025) [1].

---

## 📖 Deskripsi
Modul ini dirancang untuk memberikan pemahaman praktis dan teoretis mengenai konsep dasar hingga lanjutan dalam Pembelajaran Mesin (Machine Learning) [1]. Pendekatan pembelajaran mencakup siklus lengkap dari pemrosesan data hingga pembangunan arsitektur jaringan saraf tiruan tingkat lanjut [2, 3].

---

## 🛠️ Persyaratan Sistem & Library (Dependencies)
Berdasarkan skrip program di dalam modul ini, pastikan Anda telah menginstal beberapa *library* Python berikut:
* `pandas` & `numpy` (Untuk manipulasi data) [4, 5]
* `matplotlib` & `seaborn` (Untuk visualisasi data) [5, 6]
* `scikit-learn` (Untuk algoritma ML klasik, preprocessing, dan metrik evaluasi) [4, 6, 7]
* `torch` (PyTorch - untuk implementasi RNN) [8]
* `tensorflow` / `keras` (Untuk implementasi CNN) [9]

---

## 📚 Silabus & Outline Materi

### 🟢 Pertemuan 1: Machine Learning Workflow [2]
Fokus pada alur kerja sistematis dalam membangun model ML.
* **Definisi Masalah:** Menentukan tujuan (klasifikasi, regresi, clustering) dan metrik evaluasi (Akurasi, RMSE, F1-Score, AUC-ROC) [10, 11].
* **Pengumpulan Data:** Mengambil data dari Database, API, atau file CSV/Excel [12, 13].
* **Exploratory Data Analysis (EDA) & Preprocessing:** Penanganan *missing values*, *feature scaling*, dan *encoding* kategorikal [14-16].
* **Feature Engineering:** Transformasi dan ekstraksi fitur (misal PCA) [17].
* **Pemilihan, Pelatihan, & Evaluasi Model:** Splitting data, *cross-validation*, dan *confusion matrix* [18-20].
* **Tuning Hyperparameter & Deployment:** Penggunaan `GridSearchCV` / `RandomizedSearchCV`, penyimpanan model ke `.pkl`, dan monitoring [21, 22].

### 🟢 Pertemuan 2: Data Processing, Data Storage, Data Privacy, & Data Quality [23]
Fundamental penyiapan data skala industri sebelum masuk ke model ML.
* **Data Processing:** Ekstraksi, integrasi, dan transformasi data mentah [24, 25].
* **Data Storage:** Penyimpanan via *Flat files* (CSV/JSON), Basis Data Relasional (RDBMS), NoSQL, dan penyimpanan terdistribusi (HDFS) [26-28].
* **Data Privacy:** Praktik PPDM, Anonimisasi/Pseudonimisasi, enkripsi, dan penghapusan data [29, 30].
* **Data Quality & Cleaning:** Menjaga kelengkapan, konsistensi, keakuratan, dan validitas data, serta deteksi *outliers* [30-33].

### 🟢 Pertemuan 3: Support Vector Machine (SVM) [34]
Algoritma pencarian batas keputusan terbaik (*decision boundary*).
* **Maksimalisasi Margin & Hyperplane:** Memisahkan data dengan jarak klasifikasi maksimal [34, 35].
* **Teknik Kernel:** Menangani data non-linear (Linear, Radial Basis Function/RBF, Polynomial) [36, 37].
* **Kontrol Parameter C:** Mengatasi *Overfitting* (C besar) dan *Underfitting* (C kecil) [38, 39].
* **Implementasi:** Klasifikasi menggunakan modul `sklearn.svm.SVC` untuk data linear dan bulan (`make_moons`) [5, 7, 40].

### 🟢 Pertemuan 4: Neural Networks Dasar (Perceptron, Adaline, MLP) [41]
Pengenalan terhadap Jaringan Saraf Tiruan (*Artificial Neural Networks*).
* **Perceptron:** *Single-layer neural network* untuk klasifikasi linear dengan fungsi aktivasi *step function* [42].
* **Adaline (Adaptive Linear Neuron):** Pembelajaran menggunakan fungsi linier dan optimasi meminimalkan *Mean Squared Error* (MSE) berbasis *gradient descent* [43, 44].
* **Multilayer Perceptron (MLP):** Jaringan saraf dengan *hidden layer*, memproses data non-linear menggunakan algoritma *backpropagation* (lapisan input, *hidden*, dan *output*) [45].

### 🟢 Pertemuan 5: Learning Vector Quantization (LVQ) [46]
Klasifikasi berbasis pola kedekatan dan *prototype vectors*.
* **Konsep Prototipe:** Representasi kode perwakilan ruang fitur masing-masing kelas [46].
* **Update Rule:** Menarik prototipe mendekati data (jika kelas sama) atau menjauhkannya (jika kelas berbeda) berdasarkan perhitungan *Learning Rate* ($\alpha$) [47, 48].

### 🟢 Pertemuan 6: Radial Basis Function Neural Networks (RBFNN) [49]
Jaringan dengan fungsi aktivasi basis radial (Gaussian).
* **Arsitektur RBF:** Menggunakan jarak terhadap pusat klaster (center) pada *hidden layer* alih-alih *backpropagation* konvensional [50].
* **Fungsi Aktivasi Gaussian:** Mengontrol lebar kurva (parameter *gamma*) [51].
* **Implementasi Dua Tahap:** Menemukan *centers* dengan K-Means clustering, kemudian melatih lapisan output dengan *Linear/Ridge Regression* [52-54].

### 🟢 Pertemuan 7: Recurrent Neural Networks (RNN) [55]
Pemrosesan data sekuensial yang bergantung pada waktu (Time-Series).
* **Konsep *Hidden State*:** Mewariskan memori dari langkah/waktu sebelumnya ke langkah berikutnya [55, 56].
* **Vanishing Gradient:** Tantangan utama dalam RNN standar dan solusinya melalui arsitektur LSTM (*Long Short-Term Memory*) dan GRU [57].
* **Implementasi PyTorch:** Prediksi deret waktu berdasar data gelombang sinusoidal [8, 58].

### 🟢 Pertemuan 8: Convolutional Neural Networks (CNN) [59]
Deep learning untuk memproses gambar dan *Computer Vision*.
* **Konsep CNN:** Ekstraksi fitur otomatis dan reduksi dimensi (invarian skala/posisi) tanpa ekstraksi fitur manual [3, 60, 61].
* **Arsitektur Utama:** *Input Layer*, *Convolutional Layer* (Filter/Kernel + ReLU), *Pooling Layer* (Max/Average Pooling), *Fully Connected Layer* (Flatten), dan *Output Layer* (Softmax) [62-64].
* **Implementasi Keras/TensorFlow:** Pengklasifikasian gambar *grayscale* angka tulisan tangan (Dataset MNIST) [9, 65, 66].

### 🟢 Pertemuan 9: Boltzmann Machine & Hidden Markov Model [67]
Model probabilistik untuk data tak terawasi dan sekuensial.
* **Boltzmann Machine (BM):** Jaringan saraf stokastik berbasis energi untuk pemodelan distribusi probabilistik tanpa label [68, 69].
* **Restricted Boltzmann Machine (RBM):** Varian BM efisien yang memisahkan *Visible Units* dan *Hidden Units* tanpa koneksi intra-layer [70, 71].
* **Hidden Markov Model (HMM):** Model statistik sekuensial yang melibatkan *Hidden States*, Matriks Transisi, Matriks Emisi, dan Observasi [72, 73].

---

## 🚀 Cara Penggunaan
1. **Clone repositori ini:** `git clone <link-repo-anda>`
2. **Instalasi library:** Disarankan menggunakan _Virtual Environment_.
   ```bash
   pip install -r requirements.txt
Buka modul Jupyter Notebook (.ipynb) atau script Python (.py) di masing-masing branch Pertemuan_X untuk melihat kode dan menjalankan latihan secara interaktif.
