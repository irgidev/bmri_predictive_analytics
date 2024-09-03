# Laporan Proyek Machine Learning - Irgi Setiawan

## Domain Proyek

Investasi di pasar saham menawarkan peluang besar namun disertai risiko tinggi karena volatilitas harga. Investor membutuhkan prediksi yang akurat untuk mengurangi risiko dan meningkatkan pengambilan keputusan. Teknologi machine learning, khususnya model Long Short-Term Memory (LSTM), efektif dalam menganalisis data historis untuk memprediksi harga saham di masa depan.

Proyek ini bertujuan membangun model LSTM untuk memprediksi harga saham. Diharapkan, model ini membantu investor membuat keputusan lebih baik dengan memanfaatkan data historis untuk memprediksi pergerakan harga dalam jangka pendek.

**Rubrik/Kriteria Tambahan (Opsional)**:
Prediksi harga saham adalah tantangan penting karena volatilitas yang bisa menyebabkan kerugian signifikan. Dengan LSTM, pola-pola dalam data historis yang sulit diidentifikasi metode konvensional dapat ditangkap. Ini membantu mengurangi risiko dan memberi keunggulan kompetitif bagi investor dalam pengambilan keputusan.

Referensi Terkait:
- Brownlee, Jason. *Deep Learning for Time Series Forecasting.* Machine Learning Mastery, 2018.
- Zhang, G., Patuwo, B.E., & Hu, M.Y. *Forecasting with artificial neural networks: The state of the art.* International Journal of Forecasting, 14(1), 35-62, 1998.

## Business Understanding

### Problem Statements

- Bagaimana memprediksi harga penutupan saham BMRI.JK dalam 30 hari ke depan dengan akurasi yang memadai?
- Bagaimana memastikan model bekerja baik pada data historis dan mampu melakukan prediksi akurat pada data baru?
- Apakah model bisa memberikan keunggulan kompetitif bagi investor dalam pengambilan keputusan?

### Goals

- Mengembangkan model LSTM yang akurat untuk memprediksi harga penutupan saham BMRI.JK dalam 30 hari ke depan.
- Mengevaluasi model dengan metrik RMSE untuk memastikan kinerja pada data baru dan mencegah overfitting.
- Menyediakan alat bantu bagi investor untuk mengantisipasi pergerakan pasar dan membuat keputusan yang lebih baik.

**Rubrik/Kriteria Tambahan (Opsional)**:

### Solution Statements

- Mengembangkan model LSTM dengan dua lapisan dan 50 unit di setiap lapisan, menggunakan Dropout untuk mengurangi overfitting.
- Menggunakan prediksi dari model LSTM untuk memprediksi harga saham BMRI.JK selama 30 hari ke depan, dengan mengevaluasi performa model menggunakan metrik RMSE.

## Data Understanding
Data yang digunakan dalam proyek ini adalah data harga saham BMRI.JK yang diambil dari datasets pada website kaggle yang selalu terupdate. Dataset ini mencakup harga penutupan saham harian dari periode tertentu.

- **Jumlah Data**: Dataset ini terdiri dari 1.387 baris dan 7 kolom.
- **Kondisi Data**:
  - **Missing Values**: Tidak terdapat missing values dalam dataset ini. Setiap kolom memiliki data yang lengkap.
  - **Duplicate Values**: Tidak ditemukan duplikasi baris dalam dataset ini. Semua baris data adalah unik.

### Sumber Data

Dataset ini diambil dari sumber yang mencakup data historis harga saham BMRI.JK yang selalu di update. Data dapat diakses melalui tautan berikut:

- **Tautan Sumber Data**: [Kaggle - BMRI.JK](https://www.kaggle.com/datasets/caesarmario/bank-mandiri-stock-historical-price)

### Uraian Fitur pada Dataset

Berikut adalah uraian dari seluruh fitur yang terdapat pada dataset:

- **Date**: Tanggal pencatatan harga saham.
- **Open**: Harga pembukaan saham BMRI.JK pada hari tersebut.
- **High**: Harga tertinggi saham BMRI.JK pada hari tersebut.
- **Low**: Harga terendah saham BMRI.JK pada hari tersebut.
- **Close**: Harga penutupan saham BMRI.JK pada hari tersebut.
- **Adj Close**: Harga penutupan saham yang telah disesuaikan (adjusted close price) untuk kejadian korporasi seperti dividen, stock split, dll.
- **Volume**: Jumlah volume saham yang diperdagangkan pada hari tersebut.

**Rubrik/Kriteria Tambahan (Opsional)**:
Teknik visualisasi data digunakan untuk memahami tren umum dalam data. Misalnya, plotting harga penutupan saham terhadap waktu untuk melihat tren jangka panjang.

## Data Preparation
Pada tahap ini, data yang digunakan disiapkan untuk proses modeling. Langkah-langkah yang dilakukan termasuk:

1. **Normalisasi Data**: Menggunakan MinMaxScaler untuk menormalkan data harga saham ke rentang [0, 1] untuk membantu konvergensi model LSTM.
2. **Pemisahan Data**: Membagi data menjadi set pelatihan dan set pengujian. Set pelatihan digunakan untuk melatih model, sedangkan set pengujian digunakan untuk mengevaluasi kinerja model.
3. **Pembuatan Sequences**: Membuat sequences data untuk digunakan sebagai input dalam model LSTM. Setiap sequence terdiri dari `time_step` data sebelumnya untuk memprediksi data ke depan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Normalisasi dilakukan karena LSTM lebih cepat konvergen saat input berada dalam skala yang lebih kecil.
- Pemisahan data dilakukan untuk mengevaluasi model secara obyektif.

## Modeling
Pada tahap ini, saya mengembangkan model prediksi harga saham menggunakan Long Short-Term Memory (LSTM). LSTM adalah jenis Recurrent Neural Network (RNN) yang efektif dalam menangani data sekuensial dengan dependensi jangka panjang, seperti data time series harga saham.

### Algoritma yang Digunakan: LSTM
LSTM dipilih karena kemampuannya dalam menangani masalah vanishing gradient dan menangkap dependensi jangka panjang yang ada dalam data harga saham. Setiap sel LSTM memiliki mekanisme khusus, seperti gate input, gate forget, dan gate output, yang memungkinkan informasi penting disimpan dan digunakan untuk prediksi di masa depan.

### 1. Pengembangan Model LSTM

LSTM adalah jenis Recurrent Neural Network (RNN) yang dirancang untuk memproses, memprediksi, dan menganalisis data time series. LSTM memiliki keunggulan dalam menangani dependensi jangka panjang dalam data sekuensial, yang sangat penting dalam analisis harga saham yang dipengaruhi oleh banyak faktor historis.

#### Arsitektur Model

Model LSTM yang dikembangkan terdiri dari dua lapisan LSTM berturut-turut, masing-masing dengan 50 unit. Berikut adalah langkah-langkah dalam pengembangan model:

- **Lapisan LSTM**: Saya menggunakan dua lapisan LSTM, masing-masing dengan 50 unit. Setiap lapisan ini dilengkapi dengan mekanisme khusus yang disebut 'gates' (input gate, forget gate, dan output gate) yang memungkinkan model untuk menyimpan dan memproses informasi dari data historis yang relevan.
- **Dropout**: Untuk mengurangi overfitting, saya menambahkan dropout dengan rate 0.2 setelah setiap lapisan LSTM. Dropout secara acak menonaktifkan sebagian neuron selama pelatihan untuk mencegah model menjadi terlalu spesifik terhadap data training.
- **Dense Layer**: Lapisan Dense digunakan setelah lapisan LSTM untuk menghasilkan prediksi akhir. Ini adalah lapisan penuh yang menghubungkan semua neuron dari lapisan sebelumnya untuk membuat output akhir.
- **Output Layer**: Lapisan ini menghasilkan prediksi nilai harga saham di masa depan.

### 2. Pelatihan Model

Setelah arsitektur model LSTM selesai, model dilatih menggunakan data historis harga saham BMRI.JK. Saya membagi dataset menjadi bagian training dan testing untuk memastikan model dapat generalisasi dengan baik.

- **Training**: Dataset digunakan untuk melatih model selama sejumlah epoch tertentu, dalam hal ini 100 epochs, dengan batch size 32. Proses ini bertujuan untuk meminimalkan error antara prediksi model dan data sebenarnya melalui optimasi fungsi loss, yaitu Mean Squared Error (MSE).
- **Testing**: Setelah pelatihan, model diuji menggunakan data yang belum pernah dilihat sebelumnya untuk mengevaluasi performanya. Hasil prediksi pada data testing dibandingkan dengan nilai sebenarnya untuk menghitung RMSE (Root Mean Squared Error).

### 3. Penggunaan Callback EarlyStopping

**EarlyStopping** adalah callback yang sangat berguna dalam pelatihan model, terutama untuk mencegah overfitting dan mempercepat pelatihan model. Callback ini secara otomatis menghentikan pelatihan jika performa model pada data validasi tidak membaik setelah sejumlah epoch tertentu.

- **Parameter EarlyStopping**:
  - **monitor**: Parameter ini menentukan metrik yang akan dipantau, dalam hal ini `val_loss` (validasi loss).
  - **patience**: Parameter ini menentukan jumlah epoch tanpa peningkatan sebelum pelatihan dihentikan. Dalam proyek ini, nilai `patience` ditetapkan ke 10, yang berarti jika tidak ada perbaikan pada validasi loss selama 10 epoch berturut-turut, pelatihan akan dihentikan.
  - **restore_best_weights**: Jika diatur ke `True`, model akan mengembalikan bobot terbaik yang dicapai selama pelatihan. Ini memastikan bahwa model yang disimpan adalah model terbaik berdasarkan metrik yang dipantau.

### 4. Prediksi Model

Setelah model LSTM dilatih dan diuji, model digunakan untuk memprediksi harga saham BMRI.JK selama 30 hari ke depan. Proses ini melibatkan:

- Mengambil data penutupan harga saham terbaru sebagai input untuk model.
- Menggunakan model yang telah dilatih untuk menghasilkan prediksi harga saham hari berikutnya.
- Memasukkan prediksi ini kembali ke model untuk menghasilkan prediksi harga untuk hari-hari berikutnya, hingga 30 hari ke depan.

### Parameter yang Digunakan
Berikut adalah parameter utama yang digunakan dalam pengembangan model LSTM:
- **Number of Units**: 50 unit per lapisan LSTM.
- **Dropout Rate**: Dropout sebesar 0.2 diterapkan untuk mencegah overfitting.
- **Batch Size**: Ukuran batch yang digunakan adalah 32.
- **Epochs**: Model dilatih selama 100 epoch.
- **Loss Function**: Mean Squared Error (MSE) digunakan sebagai fungsi loss.
- **Optimizer**: Adam digunakan sebagai optimizer.
- **Callbacks**: Callback **EarlyStopping** digunakan untuk menghentikan pelatihan jika tidak ada peningkatan pada validasi loss setelah 10 epoch berturut-turut (`patience=10`). Parameter ini juga menggunakan `restore_best_weights=True` untuk memastikan model dengan performa terbaik disimpan.

### Justifikasi Penggunaan LSTM
LSTM dipilih karena kemampuannya yang unggul dalam memprediksi data time series seperti harga saham, di mana pola jangka panjang dan variasi musiman dapat mempengaruhi harga di masa depan. Penggunaan LSTM memungkinkan model untuk mempertimbangkan informasi historis yang relevan saat memprediksi harga di masa depan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
### Kelebihan dan Kekurangan LSTM:
- **Kelebihan**: LSTM mampu menangkap dependensi jangka panjang dalam data time series.
- **Kekurangan**: LSTM cenderung membutuhkan waktu pelatihan yang lebih lama dibandingkan model linear.

## Evaluation

### Evaluasi Model

Model LSTM yang dikembangkan telah dievaluasi menggunakan metrik **Root Mean Squared Error (RMSE)** untuk mengukur keakuratan prediksi harga saham BMRI.JK. RMSE memberikan informasi seberapa jauh prediksi model dari nilai sebenarnya, dengan memberikan bobot lebih pada error yang lebih besar.

**Hasil Evaluasi:**
- **Train RMSE**: 85.24
- **Test RMSE**: 132.80

### Analisis Dampak Terhadap Business Understanding

1. **Apakah Model Menjawab Problem Statement?**
   - **Problem Statement 1**: Model LSTM berhasil memprediksi harga penutupan saham BMRI.JK dalam 30 hari ke depan. Meskipun Test RMSE lebih tinggi dibandingkan dengan Train RMSE, model ini masih mampu memberikan prediksi yang dapat digunakan sebagai panduan bagi investor. Ini menunjukkan bahwa model menjawab problem statement yang pertama.
   - **Problem Statement 2**: Evaluasi menunjukkan bahwa model bekerja lebih baik pada data training dibandingkan data testing, mengindikasikan potensi overfitting. Meskipun demikian, model tetap memberikan wawasan yang berharga untuk data yang belum pernah dilihat sebelumnya, sehingga masih dapat dianggap menjawab problem statement kedua.
   - **Problem Statement 3**: Meskipun ada perbedaan dalam performa pada data training dan testing, model ini tetap memberikan keunggulan kompetitif bagi investor dengan memberikan panduan yang berbasis data historis, meskipun dengan keterbatasan tertentu.

2. **Apakah Model Berhasil Mencapai Goals yang Diharapkan?**
   - **Goal 1**: Model LSTM yang dikembangkan mampu memprediksi harga saham untuk 30 hari ke depan dengan hasil yang cukup baik, meskipun ada perbedaan antara performa pada data training dan testing.
   - **Goal 2**: Evaluasi menggunakan RMSE telah dilakukan, dan meskipun ada indikasi overfitting, model tetap memberikan hasil yang dapat digunakan dalam konteks prediksi saham.
   - **Goal 3**: Dengan model ini, investor dapat mengantisipasi pergerakan pasar lebih baik dan membuat keputusan investasi yang lebih baik, meskipun ada ruang untuk peningkatan, terutama dalam hal generalisasi model.

3. **Apakah Solusi Statement Berdampak?**
   - **Solusi 1 (LSTM Baseline)**: Pengembangan model LSTM baseline memberikan dasar yang kuat untuk memulai prediksi saham. Model baseline ini berhasil menunjukkan potensi LSTM dalam memprediksi data time series.
   - **Solusi 2 (Model Evaluasi)**: Evaluasi yang dilakukan memastikan bahwa model LSTM yang dapat digunakan secara efektif untuk memprediksi harga saham. Model ini memberikan hasil yang memadai dan sesuai dengan tujuan bisnis, meskipun evaluasi lebih lanjut diperlukan untuk mengoptimalkan performanya.  

**Rubrik/Kriteria Tambahan (Opsional)**: 
### RMSE Calculation:
<img width="450" alt="image" src="https://github.com/user-attachments/assets/3c57aa55-7a56-46e2-b7c6-db064100c4fd">

- RMSE dipilih karena memberikan bobot yang lebih tinggi pada error yang lebih besar, yang berguna dalam konteks prediksi harga saham di mana error besar lebih dihindari.


