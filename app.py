import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import json

st.set_page_config(page_title="Product Image Classification", page_icon="🛒", layout="centered")

# === Styling (opsional) ===
st.markdown("""
<style>
.pred-label{
  font-size: 28px; font-weight: 800; background: #F0F9FF; color:#0C4A6E;
  padding: 10px 14px; border-radius: 10px; display:inline-block;
}
.pred-sub{ font-size: 16px; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("efficientnet_full_model.keras")
    try:
        with open("classes.json", "r") as f:
            classes = json.load(f)  # list index->label
    except Exception:
        classes = None
    return model, classes

def to_model_input(image_pil, target_size=(224, 224)):
    # Perbaiki orientation dari EXIF, konversi pasti ke RGB (3 channel), dan PASTIKAN ukuran 224x224
    img = ImageOps.exif_transpose(image_pil).convert("RGB")
    # fit = resize + crop tengah agar ukuran PASTI tepat, tidak 225x225
    img = ImageOps.fit(img, target_size, method=Image.LANCZOS)
    x = np.asarray(img, dtype=np.float32)
    # Jika tetap grayscale (edge case), naikkan ke 3 channel
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    # Tambah dimensi batch
    x = x[None, ...]  # (1, 224, 224, 3)
    return x

model, classes = load_assets()

st.title("Product Image Classification (EfficientNetB0)")
img_w = st.sidebar.slider("Image preview width (px)", 160, 512, 320, 16)

uploaded_file = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    st.image(raw_image, caption="Uploaded Image", width=img_w)

    x = to_model_input(raw_image, (224, 224))

    # NOTE:
    # Jika SAAT TRAINING kamu SUDAH menanam preprocess_input di dalam arsitektur model,
    # langsung prediksi pakai x apa adanya (seperti di bawah).
    # Kalau TIDAK, gunakan baris berikut untuk preprocessing sebelum prediksi:
    # x = tf.keras.applications.efficientnet.preprocess_input(x)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    label = classes[idx] if classes else str(idx)

    st.markdown(f"<div class='pred-label'>Prediction: {label}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='pred-sub'>Confidence: {conf:.2%}</div>", unsafe_allow_html=True)
