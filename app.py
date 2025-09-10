import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import json, os

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

def create_efficientnet_model(num_classes, input_shape=(224,224,3), base_trainable=False, dropout_rate=0.2):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet',
        input_shape=input_shape, pooling='max'
    )
    base.trainable = base_trainable
    inputs = tf.keras.Input(shape=input_shape)
    # Preprocessing ditanam di graph (konsisten dgn training)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

@st.cache_resource
def load_assets():
    # 1) Load classes mapping lebih dulu (agar tahu num_classes untuk fallback)
    classes = None
    if os.path.isfile("classes.json"):
        with open("classes.json", "r") as f:
            classes = json.load(f)
    num_classes = len(classes) if classes else None

    # 2) Coba load full model
    try:
        model = tf.keras.models.load_model("efficientnet_full_model.keras")
        return model, classes
    except Exception as e:
        # 3) Fallback: bangun arsitektur + load weights (kalau tersedia)
        if os.path.isfile("efficientnet_only_weights.weights.h5") and num_classes is not None:
            st.warning("Gagal memuat model penuh. Menggunakan fallback: arsitektur EfficientNet + weights.")
            model = create_efficientnet_model(num_classes=num_classes, input_shape=(224,224,3))
            model.load_weights("efficientnet_only_weights.weights.h5")
            return model, classes
        # 4) Jika tidak bisa fallback, lempar error asli agar terlihat lognya
        raise e

def get_expected_input(model):
    """Ambil (H, W, C) dari input model."""
    ishape = model.inputs[0].shape  # TensorShape([None, H, W, C])
    h = int(ishape[1]) if ishape[1] is not None else 224
    w = int(ishape[2]) if ishape[2] is not None else 224
    c = int(ishape[3]) if ishape[3] is not None else 3
    return h, w, c

def to_model_input(image_pil, target_size, channels):
    """
    Convert PIL image -> np.ndarray batch (1,H,W,C) yang sesuai dengan ekspektasi model.
    - target_size: (H, W)
    - channels: 1 atau 3
    """
    img = ImageOps.exif_transpose(image_pil)  # perbaiki orientation
    if channels == 3:
        img = img.convert("RGB")
        img = ImageOps.fit(img, target_size, method=Image.LANCZOS)
        x = np.asarray(img, dtype=np.float32)  # (H,W,3)
    else:  # channels == 1
        img = img.convert("L")
        img = ImageOps.fit(img, target_size, method=Image.LANCZOS)
        x = np.asarray(img, dtype=np.float32)  # (H,W)
        x = x[..., None]                       # (H,W,1)
    x = x[None, ...]  # (1,H,W,C)
    return x

model, classes = load_assets()

# (Opsional) hard guard: jika model masih punya input channel 1, beri info
H, W, C = get_expected_input(model)
if C not in (1, 3):
    st.error(f"Model input channels tidak didukung: C={C}. Harus 1 (grayscale) atau 3 (RGB).")
    st.stop()

st.title("Product Image Classification (EfficientNetB0)")
img_w = st.sidebar.slider("Image preview width (px)", 160, 512, 320, 16)

uploaded_file = st.file_uploader("Upload product image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    st.image(raw_image, caption="Uploaded Image", width=img_w)

    # Resize & channel sesuai ekspektasi model (akan handle 224 vs 225, 1 vs 3)
    x = to_model_input(raw_image, (H, W), C)

    # ⚠️ Kode training kita MENANAM preprocess_input di dalam arsitektur model.
    # Jadi di inference TIDAK perlu memanggil preprocess_input lagi di sini.
    # Jika versi model Anda TIDAK menanam preprocessing, aktifkan baris berikut:
    # x = tf.keras.applications.efficientnet.preprocess_input(x)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    label = classes[idx] if classes else str(idx)

    st.markdown(f"<div class='pred-label'>Prediction: {label}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='pred-sub'>Confidence: {conf:.2%}</div>", unsafe_allow_html=True)
