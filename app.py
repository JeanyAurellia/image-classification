import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

st.set_page_config(page_title="Image Classification App", page_icon="🖼️", layout="centered")

# ==== CSS: perbesar teks hasil prediksi ====
st.markdown("""
<style>
.pred-label{
  font-size: 28px; 
  font-weight: 800; 
  background: #F0F9FF; 
  color:#0C4A6E;
  padding: 10px 14px; 
  border-radius: 10px; 
  display:inline-block;
}
.pred-sub{
  font-size: 16px; 
  opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("efficientnet_full_model.keras")
    # jika kamu punya classes.json dari training
    try:
        with open("classes.json", "r") as f:
            classes = json.load(f)   # list index -> label
    except Exception:
        classes = None  # fallback: tidak ada mapping
    return model, classes

model, classes = load_assets()

st.title("Image Classification App")

# Opsional: slider untuk ngatur lebar preview gambar
img_w = st.sidebar.slider("Image preview width (px)", 160, 512, 320, 16)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=img_w)  # << kecilkan lewat width, bukan use_column_width

    # Preprocess: model kita sudah menanam preprocess_input di graph, jadi tanpa /255.0
    img = image.resize((224, 224))
    x = np.array(img, dtype=np.float32)[None, ...]  # shape (1,224,224,3)

    # Jika model-mu TIDAK menanam preprocess_input:
    # x = tf.keras.applications.efficientnet.preprocess_input(np.array(img, dtype=np.float32))[None, ...]

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    label = classes[idx] if classes else str(idx)  # jika tidak ada classes.json, tampilkan index

    # Hasil prediksi dengan font besar
    st.markdown(f"<div class='pred-label'>Prediction: {label}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='pred-sub'>Confidence: {conf:.2%}</div>", unsafe_allow_html=True)

    # # (opsional) tampilkan Top-3
    # if classes:
    #     st.subheader("Top-3 probabilities")
    #     top3 = probs.argsort()[-3:][::-1]
    #     for i in top3:
    #         st.write(f"- {classes[int(i)]}: {probs[int(i)]:.2%}")
