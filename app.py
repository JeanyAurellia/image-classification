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
    # Determine if this should be grayscale or RGB based on input_shape
    input_channels = input_shape[2]
    
    if input_channels == 1:
        # For grayscale, we need to build without pre-trained weights
        # since ImageNet weights are for RGB (3 channels)
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, 
            weights=None,  # No pre-trained weights for grayscale
            input_shape=input_shape, 
            pooling='max'
        )
    else:
        # For RGB, use ImageNet weights
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, 
            weights='imagenet',
            input_shape=input_shape, 
            pooling='max'
        )
    
    base.trainable = base_trainable
    inputs = tf.keras.Input(shape=input_shape)
    
    # Only apply EfficientNet preprocessing for RGB images
    if input_channels == 3:
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    else:
        # For grayscale, just normalize to [0,1] or [-1,1] depending on your training
        x = inputs / 255.0  # Adjust this based on how you preprocessed during training
    
    x = base(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def detect_model_input_shape():
    """
    Try to detect the correct input shape by examining available weight files
    or use a heuristic approach
    """
    # First, try to load the full model to get the correct shape
    try:
        temp_model = tf.keras.models.load_model("efficientnet_full_model.keras")
        return temp_model.input_shape[1:]  # Remove batch dimension
    except:
        pass
    
    # If we have weights file, we can try to inspect it
    # This is a fallback - you might need to adjust based on your actual model
    if os.path.isfile("efficientnet_only_weights.weights.h5"):
        # Based on your error, it seems your model expects 1 channel (grayscale)
        # Adjust this if your model actually expects different dimensions
        return (224, 224, 1)  # Grayscale
    
    # Default fallback
    return (224, 224, 3)  # RGB

@st.cache_resource
def load_assets():
    # 1) Load classes mapping
    classes = None
    if os.path.isfile("classes.json"):
        with open("classes.json", "r") as f:
            classes = json.load(f)
    num_classes = len(classes) if classes else None

    # 2) Try to load full model first
    try:
        model = tf.keras.models.load_model("efficientnet_full_model.keras")
        st.success("✅ Full model loaded successfully!")
        return model, classes
    except Exception as e:
        error_msg = str(e)
        st.info(f"Could not load full model: {error_msg[:200]}...")
        
        # Extract input shape from error message
        detected_shape = None
        if "225, 225, 1" in error_msg:
            detected_shape = (225, 225, 1)
            st.info("🔍 Detected from error: Model expects (225, 225, 1)")
        elif "224, 224, 3" in error_msg:
            detected_shape = (224, 224, 3)
            st.info("🔍 Detected from error: Model expects (224, 224, 3)")
        elif "224, 224, 1" in error_msg:
            detected_shape = (224, 224, 1)
            st.info("🔍 Detected from error: Model expects (224, 224, 1)")
        
        # 3) Try with detected shape first
        if detected_shape and os.path.isfile("efficientnet_only_weights.weights.h5") and num_classes:
            st.warning(f"Trying to build model with detected shape: {detected_shape}")
            
            try:
                model = create_efficientnet_model(
                    num_classes=num_classes, 
                    input_shape=detected_shape
                )
                model.load_weights("efficientnet_only_weights.weights.h5")
                st.success("✅ Model built with detected architecture!")
                return model, classes
            except Exception as arch_error:
                st.warning(f"Detected architecture failed: {str(arch_error)[:150]}...")
        
        # 4) Fallback: test weights file with different architectures
        if os.path.isfile("efficientnet_only_weights.weights.h5") and num_classes:
            st.warning("Testing different architectures to match weights file...")
            
            weights_shape = detect_weights_input_shape(num_classes)
            if weights_shape:
                st.success(f"Found compatible architecture: {weights_shape}")
                model = create_efficientnet_model(
                    num_classes=num_classes, 
                    input_shape=weights_shape
                )
                model.load_weights("efficientnet_only_weights.weights.h5")
                st.success("✅ Model built with weights-compatible architecture!")
                return model, classes
        
        # 5) Last resort - provide clear guidance
        st.error("❌ Could not load model with any architecture.")
        st.error("**Your files seem to be from different training sessions:**")
        st.error(f"• Full model (.keras): Expects input shape around (225, 225, 1)")
        st.error(f"• Weights file (.h5): Seems to be from a different model")
        st.error("")
        st.error("**Solutions:**")
        st.error("1. Use model files from the same training session")
        st.error("2. Or delete one file type and use only the other")
        st.error("3. Re-train your model to ensure consistency")
        raise e

def get_expected_input(model):
    """Get (H, W, C) from model input."""
    ishape = model.inputs[0].shape  # TensorShape([None, H, W, C])
    h = int(ishape[1]) if ishape[1] is not None else 224
    w = int(ishape[2]) if ishape[2] is not None else 224
    c = int(ishape[3]) if ishape[3] is not None else 3
    return h, w, c

def to_model_input(image_pil, target_size, channels):
    """
    Convert PIL image -> np.ndarray batch (1,H,W,C) for model input.
    - target_size: (H, W)
    - channels: 1 or 3
    """
    img = ImageOps.exif_transpose(image_pil)  # Fix orientation
    
    if channels == 3:
        img = img.convert("RGB")
        img = ImageOps.fit(img, target_size, method=Image.LANCZOS)
        x = np.asarray(img, dtype=np.float32)  # (H,W,3)
    else:  # channels == 1
        img = img.convert("L")  # Convert to grayscale
        img = ImageOps.fit(img, target_size, method=Image.LANCZOS)
        x = np.asarray(img, dtype=np.float32)  # (H,W)
        x = x[..., None]                       # (H,W,1)
    
    x = x[None, ...]  # (1,H,W,C)
    return x

# Load model and classes
try:
    model, classes = load_assets()
    
    # Get expected input dimensions
    H, W, C = get_expected_input(model)
    
    if C not in (1, 3):
        st.error(f"Unsupported input channels: C={C}. Must be 1 (grayscale) or 3 (RGB).")
        st.stop()
    
    st.success(f"Model loaded successfully! Expected input: {H}×{W}×{C}")
    
    # UI
    st.title("Product Image Classification (EfficientNetB0)")
    img_w = st.sidebar.slider("Image preview width (px)", 160, 512, 320, 16)
    
    if C == 1:
        st.info("📸 Model expects **grayscale** images")
    else:
        st.info("🌈 Model expects **RGB** images")

    uploaded_file = st.file_uploader("Upload product image", type=["jpg","jpeg","png"])
    
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
        st.image(raw_image, caption="Uploaded Image", width=img_w)

        # Prepare input according to model expectations
        x = to_model_input(raw_image, (H, W), C)
        
        # Make prediction
        with st.spinner("Classifying..."):
            probs = model.predict(x, verbose=0)[0]
            idx = int(np.argmax(probs))
            conf = float(np.max(probs))
            label = classes[idx] if classes else str(idx)

        # Display results
        st.markdown(f"<div class='pred-label'>Prediction: {label}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='pred-sub'>Confidence: {conf:.2%}</div>", unsafe_allow_html=True)
        
        # Show top predictions
        if classes and len(classes) > 1:
            st.subheader("Top Predictions:")
            top_indices = np.argsort(probs)[::-1][:min(5, len(classes))]
            for i, idx in enumerate(top_indices):
                st.write(f"{i+1}. **{classes[idx]}**: {probs[idx]:.2%}")

except Exception as e:
    st.error("Failed to load the model. Please check your model files.")
    st.exception(e)
    st.stop()
