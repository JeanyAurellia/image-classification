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

def test_architecture(num_classes, input_shape):
    """Test if a specific architecture works with the weights file"""
    try:
        test_model = create_efficientnet_model(
            num_classes=num_classes, 
            input_shape=input_shape
        )
        test_model.load_weights("efficientnet_only_weights.weights.h5")
        del test_model  # Clean up
        return True
    except Exception:
        return False

def detect_weights_compatible_shape(num_classes):
    """Test different architectures to find one compatible with weights file"""
    if not os.path.isfile("efficientnet_only_weights.weights.h5"):
        return None
        
    # Test configurations in order of likelihood
    test_configs = [
        (224, 224, 3),  # Standard RGB
        (225, 225, 3),  # Higher res RGB  
        (224, 224, 1),  # Standard grayscale
        (225, 225, 1),  # Higher res grayscale
    ]
    
    st.info("🔍 Testing architectures to find weights compatibility...")
    
    for h, w, c in test_configs:
        st.info(f"Testing {h}×{w}×{c}...")
        if test_architecture(num_classes, (h, w, c)):
            st.success(f"✅ Found compatible architecture: {h}×{w}×{c}")
            return (h, w, c)
        else:
            st.warning(f"❌ {h}×{w}×{c} not compatible")
    
    return None

@st.cache_resource
def load_assets():
    # 1) Load classes mapping
    classes = None
    if os.path.isfile("classes.json"):
        with open("classes.json", "r") as f:
            classes = json.load(f)
    num_classes = len(classes) if classes else None
    
    if not num_classes:
        st.error("❌ Could not load classes.json or it's empty")
        st.stop()

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
        if detected_shape and os.path.isfile("efficientnet_only_weights.weights.h5"):
            st.warning(f"Trying to build model with detected shape: {detected_shape}")
            
            if test_architecture(num_classes, detected_shape):
                st.success("✅ Detected architecture is compatible with weights!")
                model = create_efficientnet_model(
                    num_classes=num_classes, 
                    input_shape=detected_shape
                )
                model.load_weights("efficientnet_only_weights.weights.h5")
                return model, classes
            else:
                st.warning("❌ Detected architecture not compatible with weights file")
        
        # 4) Fallback: test different architectures
        if os.path.isfile("efficientnet_only_weights.weights.h5"):
            weights_shape = detect_weights_compatible_shape(num_classes)
            if weights_shape:
                st.success(f"Building model with compatible architecture: {weights_shape}")
                model = create_efficientnet_model(
                    num_classes=num_classes, 
                    input_shape=weights_shape
                )
                model.load_weights("efficientnet_only_weights.weights.h5")
                st.success("✅ Model loaded with compatible architecture!")
                return model, classes
        
        # 5) Show clear error message
        st.error("❌ Could not load model with any architecture.")
        st.error("")
        st.error("**Diagnosis:**")
        if detected_shape:
            st.error(f"• Your .keras file expects: {detected_shape}")
        st.error("• Your .weights.h5 file seems incompatible")
        st.error("")
        st.error("**Solutions:**")
        st.error("1. **Recommended**: Delete the .weights.h5 file and use only the .keras file")
        st.error("2. **Or**: Delete the .keras file and use only the .weights.h5 file")  
        st.error("3. **Or**: Re-train to ensure both files match the same architecture")
        st.stop()

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

# Main app
try:
    # Load model and classes
    model, classes = load_assets()
    
    # Get expected input dimensions
    H, W, C = get_expected_input(model)
    
    if C not in (1, 3):
        st.error(f"Unsupported input channels: C={C}. Must be 1 (grayscale) or 3 (RGB).")
        st.stop()
    
    st.success(f"🎯 Model loaded successfully! Expected input: {H}×{W}×{C}")
    
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
    st.error("Failed to initialize the application.")
    st.exception(e)
