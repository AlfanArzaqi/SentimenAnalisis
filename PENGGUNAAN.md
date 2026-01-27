# Panduan Penggunaan - Indonesian Sentiment Analysis

## Perubahan Utama / Main Changes

### 1. Data Bahasa Indonesia / Indonesian Language Data
- **Sebelum (Before)**: Menggunakan data review berbahasa Inggris dari Play Store
- **Sesudah (After)**: Menggunakan data review berbahasa Indonesia dari Play Store Indonesia
- **Implementasi**: Parameter `lang='id'` dan `country='id'` pada fungsi scraping

### 2. Pelabelan Berbasis Konteks / Context-Based Labeling
- **Sebelum (Before)**: Label sentiment berdasarkan rating (1-2=negatif, 3=netral, 4-5=positif)
- **Sesudah (After)**: Label sentiment berdasarkan analisis teks menggunakan algoritma leksikon

#### Cara Kerja Algoritma / Algorithm Details:

1. **Leksikon Sentiment Indonesia**
   - 40+ kata positif (bagus, baik, mantap, keren, dll)
   - 40+ kata negatif (buruk, jelek, error, lambat, dll)
   - Support bahasa Inggris untuk konten campuran

2. **Penanganan Negasi / Negation Handling**
   - Kata negasi: tidak, bukan, jangan, tanpa, kurang, belum
   - Contoh: "tidak bagus" → negatif (bukan positif)

3. **Penanganan Intensifier / Intensifier Detection**
   - Kata penguat: sangat, sekali, banget, amat, super
   - Contoh: "sangat bagus" → bobot lebih tinggi

4. **Klasifikasi Berdasarkan Rasio**
   - Ratio ≥ 60% kata positif → Positif
   - Ratio ≤ 40% kata positif → Negatif
   - Lainnya → Netral

### 3. WordCloud Visualization
- **Fitur Baru**: Visualisasi distribusi kata untuk setiap kategori sentiment
- **Output**: 3 file PNG (positive, neutral, negative)
- **Lokasi**: `data/wordcloud_*.png`
- **Manfaat**: Memudahkan analisis kata-kata kunci untuk setiap sentiment

### 4. Stopwords Bahasa Indonesia
- **Fitur Baru**: 50+ stopwords bahasa Indonesia
- **Manfaat**: Text preprocessing lebih akurat untuk bahasa Indonesia
- **Support**: Tetap mendukung stopwords Inggris untuk konten campuran

## Contoh Penggunaan / Usage Examples

### Contoh Review yang Dilabel Otomatis:

```python
# Contoh Positif
"Aplikasi sangat bagus dan mudah digunakan" → POSITIVE
"Mantap banget! Recommended" → POSITIVE

# Contoh Negatif  
"Aplikasi jelek dan sering error" → NEGATIVE
"Buruk sekali, sering crash" → NEGATIVE

# Contoh dengan Negasi
"Tidak bagus, sering error" → NEGATIVE
"Tidak jelek, cukup bagus" → POSITIVE

# Contoh Netral
"Aplikasi untuk belanja online" → NEUTRAL
```

## Keuntungan Pendekatan Baru / Advantages

1. **Lebih Akurat**: Label berdasarkan konten teks aktual, bukan rating
2. **Kontekstual**: Mempertimbangkan struktur kalimat dan konteks
3. **Fleksibel**: Tidak tergantung pada sistem rating yang mungkin bias
4. **Transparan**: Dapat melihat kata-kata yang mempengaruhi klasifikasi
5. **Visual**: WordCloud membantu memahami distribusi kata per sentiment

## Cara Menjalankan / How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Jalankan Jupyter Notebook:
```bash
jupyter notebook sentiment_analysis_pipeline.ipynb
```

3. Jalankan semua cell secara berurutan

4. Lihat hasil di folder `data/`:
   - `playstore_reviews.csv` - Data mentah
   - `playstore_cleaned.csv` - Data yang sudah dibersihkan
   - `wordcloud_*.png` - Visualisasi wordcloud
   - File hasil lainnya (confusion matrix, training history, dll)

## Catatan Penting / Important Notes

- Algoritma pelabelan dapat disesuaikan dengan menambah/mengurangi kata di leksikon
- WordCloud otomatis dibuat setelah proses klasifikasi
- Data Indonesia mungkin memiliki karakteristik berbeda dari data Inggris
- Model perlu di-train ulang dengan data Indonesia untuk hasil optimal
