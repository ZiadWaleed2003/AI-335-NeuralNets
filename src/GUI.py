import streamlit as st
from PIL import Image
import tempfile
import os

from src.inference import inference

# Page configuration
st.set_page_config(
    page_title="Car Classification AI",
    page_icon="🚗",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f7ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .confidence-high {
        color: #2e7d32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f9a825;
        font-weight: bold;
    }
    .confidence-low {
        color: #c62828;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🚗 Car Classification AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image of a car and let AI identify the make and model</p>', unsafe_allow_html=True)

st.divider()

# Model selection dropdown
col1, col2 = st.columns([2, 1])
with col1:
    model_options = {
        "VGG19": "vgg",
        "ResNet50": "resnet",
        "MobileNet": "mobilenet",
        "InceptionV1": "inceptionv1"
    }
    selected_model = st.selectbox(
        "🔧 Select Model",
        options=list(model_options.keys()),
        help="Choose the deep learning model for classification"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"Using: **{selected_model}**")

st.divider()

# Image upload section
uploaded_file = st.file_uploader(
    "📤 Upload a car image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP"
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.divider()
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔍 Predict", type="primary", use_container_width=True)
    
    if predict_button:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            # Convert RGBA to RGB if necessary (for PNG with transparency)
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                rgb_image.save(tmp_file.name)
            else:
                image.convert('RGB').save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            with st.spinner("🔄 Analyzing image..."):
                # Get the model key for inference
                model_key = model_options[selected_model]
                
                # Run inference
                result = inference(model_key, tmp_path)
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            st.markdown("### 🎯 Prediction Result")
            
            # Main prediction
            confidence = result['confidence']
            if confidence >= 0.7:
                conf_class = "confidence-high"
            elif confidence >= 0.4:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"
            
            st.markdown(f"""
            <div class="result-box">
                <h3 style="margin:0; color:#1E88E5;">🚙 {result['predicted_class_name']}</h3>
                <p style="margin:10px 0 0 0;">
                    Confidence: <span class="{conf_class}">{confidence * 100:.2f}%</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Top 5 predictions
            st.markdown("### 📊 Top 5 Predictions")
            
            for i, pred in enumerate(result['top5_predictions'], 1):
                conf = pred['confidence']
                st.markdown(f"""
                **{i}. {pred['class_name']}**  
                Confidence: `{conf * 100:.2f}%`
                """)
                st.progress(conf)
        
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.exception(e)
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

else:
    # Placeholder when no image is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 50px; background-color: #f5f5f5; border-radius: 10px; border: 2px dashed #ccc;">
        <p style="font-size: 1.2rem; color: #888;">📷 Upload an image to get started</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    <p>Built with ❤️ using Streamlit | Car Classification using Deep Learning</p>
    <p>Models: VGG19, ResNet50, MobileNet, InceptionV1</p>
</div>
""", unsafe_allow_html=True)
