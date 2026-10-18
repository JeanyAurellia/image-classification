# 🖼️ Image Classification — EfficientNetB0 + Streamlit

Aplikasi **Streamlit** untuk mengklasifikasikan **gambar barang/produk** menggunakan **EfficientNetB0 (TensorFlow/Keras)**.
Project ini dibuat untuk **Praktikum AI** guna memenuhi **mata kuliah Artificial Intelligence (AI)**.

---

## 🔗 Live App

👉 **Buka aplikasinya di sini:** **[Open the app](https://image-classification-thing.streamlit.app/)**

---

## 🚀 Cara Pakai (di Web)

1. Klik tautan **Live App** di atas.
2. **Upload** satu gambar produk (format: `.jpg`, `.jpeg`, atau `.png`).
3. Tunggu proses inferensi (beberapa detik).
4. Lihat **label prediksi**, **confidence**, dan **Top-3 probabilities**.
5. (Opsional) Atur **lebar preview gambar** dari panel **sidebar**.

---

## ✨ Fitur Utama

* Klasifikasi **gambar barang/produk** single-image.
* Tampilkan **label** + **confidence**.
* UI ringkas; ukuran preview gambar bisa disesuaikan.
* Konsisten dengan **preprocessing EfficientNet**.

---

## 🧠 Model & Data (Singkat)

* Arsitektur: **EfficientNetB0** (`include_top=False` + head Dense softmax).
* Preprocessing: `tf.keras.applications.efficientnet.preprocess_input`.
* Artefak:

  * `efficientnet_full_model.keras` → **model penuh** (arsitektur + bobot).
  * `classes.json` → **mapping index → nama label** (diexport dari `train_generator.class_indices`).
* Catatan: pada skenario Kaggle umum, folder `test/` **tidak berlabel** (pakai prediksi file-by-file, bukan `evaluate()`).

---

## 🗂️ Struktur Proyek (Ringkas)

```
.
├─ app.py
├─ efficientnet_full_model.keras   # model penuh (untuk inference)
├─ classes.json                    # mapping index → label
└─ requirements.txt
```

---

## 📝 Contoh Label Output

* **Prediction:** `sneakers`
* **Confidence:** `97.4%`

> *Nama label* tergantung kelas pada dataset training Anda.

---

## 🔧 Troubleshooting (Saat Menggunakan Web App)

* **Gambar tidak terbaca** → pastikan format `.jpg/.jpeg/.png` dan ukuran file wajar.
* **Prediksi tidak sesuai** → coba gambar dengan objek utama jelas, latar sederhana, dan pencahayaan baik.
* **Label berupa angka** → pastikan `classes.json` tersedia; aplikasi memetakan index ke nama label dari berkas ini.

---

## 🎯 Tujuan Akademik

Project ini dibuat untuk **Praktikum AI** dalam rangka memenuhi **mata kuliah AI**, dengan studi kasus **klasifikasi gambar barang/produk**.

