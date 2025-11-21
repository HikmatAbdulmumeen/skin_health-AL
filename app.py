import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Skin Health AI",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for styling with background color
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fa;
    }
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px solid #1f77b4;
    }
    .warning-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00cc00;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
    }
    .upload-instruction {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="skin_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Header
st.markdown('<h1 class="main-header">🔬 Skin Health AI Assistant</h1>', unsafe_allow_html=True)

# Medical Disclaimer in a prominent box
with st.container():
    st.markdown("""
    <div class="warning-box">
    <h3>⚠️ Important Medical Disclaimer</h3>
    <b>This is an AI demonstration tool for educational purposes only.</b><br>
    It is <b>NOT</b> a medical device and <b>SHOULD NOT</b> be used for medical diagnosis.<br>
    Always consult a qualified dermatologist for any skin concerns.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Upload instructions
st.markdown("""
<div class="upload-instruction">
    <h3>📸 How to Use This Tool</h3>
    <b>1.</b> Take a clear, well-lit photo of the skin area<br>
    <b>2.</b> Upload it using the button below<br>
    <b>3.</b> Wait for the AI analysis<br>
    <b>4.</b> Review the results (this is for education only)<br>
    <b>5.</b> Refresh the page to check another image
</div>
""", unsafe_allow_html=True)

# Use session state to manage the uploaded file
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# Main content in two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Skin Image")
    
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg"], key="file_uploader")
    
    if uploaded_file and not st.session_state.analyzed:
        image = Image.open(uploaded_file).resize((128, 128))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary"):
            st.session_state.analyzed = True
            st.session_state.image = image
            st.session_state.uploaded_file = uploaded_file
            st.rerun()

with col2:
    if st.session_state.get('analyzed', False):
        st.subheader("🔍 Analysis Results")
        
        image = st.session_state.image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess and predict
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        classes = ['acne', 'bkl', 'eczema', 'lupus', 'mel', 'moles', 'normal', 'nv', 'skincancer']
        result = classes[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        disease_names = {
            'acne': '🧴 Acne',
            'bkl': '🔵 Benign Keratosis', 
            'eczema': '🔴 Eczema',
            'lupus': '🟣 Lupus',
            'mel': '🎗️ Melanoma',
            'moles': '⚫ Moles',
            'normal': '✅ Normal Skin',
            'nv': '⚪ Melanocytic Nevi',
            'skincancer': '🎗️ Skin Cancer'
        }
        
        # Display prediction in a nice box
        st.markdown(f"""
        <div class="prediction-box">
            <h3>AI Prediction: {disease_names[result]}</h3>
            <h4>Confidence Level: {confidence:.2%}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Special warnings/success messages
        if result in ['mel', 'skincancer']:
            st.markdown("""
            <div class="warning-box">
                <h3>🚨 URGENT MEDICAL ATTENTION NEEDED</h3>
                This prediction indicates potential skin cancer.<br>
                <b>Please consult a healthcare professional immediately!</b>
            </div>
            """, unsafe_allow_html=True)
        elif result == 'normal':
            st.markdown("""
            <div class="success-box">
                <h3>✅ No Concerning Conditions Detected</h3>
                The AI doesn't detect any concerning skin conditions in this image.
            </div>
            """, unsafe_allow_html=True)
        
        # Clear analysis section
        st.markdown("---")
        st.markdown("""
        <div class="info-box">
            <h3>🔄 Want to check another image?</h3>
            <b>To analyze a different skin image:</b><br>
            • Refresh the page (F5 or browser refresh button)<br>
            • Or click the button below to restart
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Start Over with New Image"):
            st.session_state.analyzed = False
            st.session_state.pop('image', None)
            st.session_state.pop('uploaded_file', None)
            st.rerun()
        
        # Show all probabilities in an expander
        with st.expander("📊 View Detailed Analysis Probabilities"):
            for i, cls in enumerate(classes):
                prob = prediction[0][i]
                st.write(f"{disease_names[cls]}: {prob:.2%}")
                st.progress(float(prob))

# Footer
st.markdown("---")
st.markdown("**For educational purposes only**")