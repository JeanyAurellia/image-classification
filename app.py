import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import json

st.set_page_config(page_title="Image Classification", page_icon="🛒", layout="centered")

# (opsional) styling
st.markdown("""
<style>
.pred-label{font-size:28px;font-weight:800;background:#F0F9FF;color:#0C4A6E;
padding:10px 14px;border-radius:10px;display:inline-block;}
.pred-sub{font-size:16px;opacity:.85;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    # kalau kamu rename, pakai nama yang baru
    model = tf.keras.models.load_model("efficientnet_full_model_FIXED.keras", compile=False)
    with open("classes.json","r") as f:
        classes = json.load(f)
    return model, classes

model, classes = load_assets()

st.title("Product Image Classification (EfficientNetB0)")
img_w = st.sidebar.slider("Image preview width (px)", 160, 512, 320, 16)

uploaded = st.file_uploader("Upload product image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=img_w)

    # Wrapper mengharapkan (224x224x3)
    img224 = ImageOps.fit(img, (224,224), method=Image.LANCZOS)
    x = np.asarray(img224, dtype=np.float32)[None, ...]  # (1,224,224,3)

    # Tidak perlu preprocessing lagi (sudah ditangani di dalam wrapped/original)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    label = classes[idx] if classes else str(idx)

    st.markdown(f"<div class='pred-label'>Prediction: {label}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='pred-sub'>Confidence: {conf:.2%}</div>", unsafe_allow_html=True)

    if classes and len(classes) > 1:
        st.subheader("Top-3 probabilities")
        top3 = probs.argsort()[-3:][::-1]
        for i in top3:
            st.write(f"- {classes[int(i)]}: {probs[int(i)]:.2%}")
