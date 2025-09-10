import os, json, zipfile
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ===============================
# Page config & simple styling
# ===============================
st.set_page_config(page_title="Product Image Classification", page_icon="🛒", layout="centered")
st.markdown("""
<style>
.pred-label{font-size:28px;font-weight:800;background:#F0F9FF;color:#0C4A6E;
padding:10px 14px;border-radius:10px;display:inline-block;}
.pred-sub{font-size:16px;opacity:.85;}
</style>
""", unsafe_allow_html=True)

# ==================================================
# Konfigurasi nama file model & classes.json
# UBAH jika nama/letaknya berbeda di repo Anda.
# ==================================================
PREFERRED_MODEL = "efficientnet_full_model_FIXED.keras"  # ganti jika perlu
MODEL_CANDIDATES = [
    PREFERRED_MODEL,
    "efficientnet_full_model.keras",
    "efficientnet_full_model_WRAPPED.keras",
    "models/efficientnet_full_model_FIXED.keras",
    "models/efficientnet_full_model.keras",
    "models/efficientnet_full_model_WRAPPED.keras",
]
CLASSES_CANDIDATES = ["classes.json", "models/classes.json"]

# ==================================================
# Util: cek apakah file .keras valid & lihat inputnya
# ==================================================
def is_git_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(200)
        return b"git-lfs" in head
    except Exception:
        return False

def inspect_keras_config(path: str):
    """Kembalikan batch_input_shape dari config.json di dalam .keras (jika ada)."""
    try:
        with zipfile.ZipFile(path, "r") as z:
            cfg = json.loads(z.read("config.json"))
        in_layers = [L for L in cfg.get("layers", []) if L.get("class_name") == "InputLayer"]
        if in_layers:
            return in_layers[0]["config"].get("batch_input_shape")
    except Exception:
        pass
    return None

def find_first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

# ==================================================
# Data prep sesuai input model (H,W,C)
# ==================================================
def to_model_input(image_pil: Image.Image, target_size, channels: int):
    """
    Convert PIL -> (1,H,W,C) float32 sesuai ekspektasi model.
    """
    img = ImageOps.exif_transpose(image_pil)
    if channels == 3:
        img = img.convert("RGB")
    else:
        img = img.convert("L")
    img = ImageOps.fit(img, target_size, method=Image.LANCZOS)
    x = np.asarray(img, dtype=np.float32)
    if channels == 1 and x.ndim == 2:
        x = x[..., None]  # (H,W,1)
    x = x[None, ...]      # (1,H,W,C)
    return x

# ==================================================
# Load assets (model .keras + classes.json)
# ==================================================
@st.cache_resource
def load_assets():
    # --- cari file model ---
    model_path = find_first_existing(MODEL_CANDIDATES)
    if not model_path:
        st.error("❌ File model .keras tidak ditemukan di repo. Pastikan nama & lokasinya benar.")
        st.write("CWD:", os.getcwd())
        st.write("Files:", os.listdir("."))
        st.stop()

    # --- validasi dasar ---
    size = os.path.getsize(model_path)
    st.caption(f"Found model file: `{model_path}` ({size} bytes)")
    if size < 1_000_000 or is_git_lfs_pointer(model_path):
        st.error("❌ File model terlihat kecil / pointer Git LFS. Pastikan yang di-commit adalah file .keras asli (bukan pointer).")
        st.stop()

    # --- intip config .keras untuk melihat input shape yang terekam ---
    recorded = inspect_keras_config(model_path)
    if recorded:
        st.caption(f"Recorded batch_input_shape in .keras: {recorded}")
    else:
        st.caption("Could not read batch_input_shape from config.json (ok, lanjut).")

    # --- coba load model penuh ---
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success(f"✅ Loaded full model: `{model_path}`")
    except Exception as e:
        st.error("❌ Gagal load model .keras. Ini biasanya terjadi kalau file korup/inkonsisten "
                 "(mis. input 1-channel tapi backbone 3-channel).")
        st.exception(e)
        st.stop()

    # --- load classes.json ---
    classes_path = find_first_existing(CLASSES_CANDIDATES)
    if not classes_path:
        st.error("❌ `classes.json` tidak ditemukan. Taruh berdampingan dengan model atau di folder `models/`.")
        st.write("Files:", os.listdir("."))
        st.stop()

    try:
        with open(classes_path, "r") as f:
            classes = json.load(f)
    except Exception as e:
        st.error("❌ Gagal membaca classes.json.")
        st.exception(e)
        st.stop()

    # --- sanity check classes ---
    if not isinstance(classes, list) or len(classes) == 0:
        st.error("❌ classes.json tidak valid (harus list label).")
        st.stop()

    return model, classes

# ===============================
# App starts here
# ===============================
model, classes = load_assets()

# Ambil ekspektasi input (H,W,C) dari model yang ter-load
ishape = model.inputs[0].shape  # (None, H, W, C)
H = int(ishape[1]) if ishape[1] is not None else 224
W = int(ishape[2]) if ishape[2] is not None else 224
C = int(ishape[3]) if ishape[3] is not None else 3
st.caption(f"Model expects input: (H,W,C)=({H},{W},{C})")

st.title("Product Image Classification (EfficientNetB0)")
img_w = st.sidebar.slider("Image preview width (px)", 160, 512, 320, 16)

uploaded = st.file_uploader("Upload product image", type=["jpg","jpeg","png"])
if uploaded:
    raw = Image.open(uploaded)
    st.image(raw, caption="Uploaded Image", width=img_w)

    x = to_model_input(raw, (H, W), C)

    # Penting: training-mu sebelumnya menanam preprocess di dalam graph,
    # jadi TIDAK perlu normalisasi lagi di sini.
    with st.spinner("Classifying..."):
        probs = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    label = classes[idx] if idx < len(classes) else str(idx)

    st.markdown(f"<div class='pred-label'>Prediction: {label}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='pred-sub'>Confidence: {conf:.2%}</div>", unsafe_allow_html=True)

    if len(classes) > 1:
        st.subheader("Top-3 probabilities")
        top3 = probs.argsort()[-3:][::-1]
        for i in top3:
            st.write(f"- {classes[int(i)]}: {probs[int(i)]:.2%}")
