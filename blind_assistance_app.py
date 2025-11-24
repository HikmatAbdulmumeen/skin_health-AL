
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image
import gtts
from gtts import gTTS
import io
import os

# Set page config
st.set_page_config(page_title="Blind Assistance AI", page_icon="🦯", layout="wide")

# Medical disclaimer
st.warning(" **BLIND ASSISTANCE AI** - FOR DEMONSTRATION PURPOSES ONLY")

st.title(" Blind Assistance AI")
st.write("Take a photo and this AI will describe what it sees to assist visually impaired users")

# Load model (cached)
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

model = load_model()

# Object classes
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'chair', 
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop'
]

def detect_objects(image):
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = image_tensor[tf.newaxis, ...]
    detections = model(image_tensor)
    return detections

def create_description(detections):
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()
    
    object_counts = {}
    for i in range(len(scores)):
        if scores[i] > 0.5 and classes[i] < len(CLASS_NAMES):
            obj_name = CLASS_NAMES[classes[i]]
            object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
    
    descriptions = []
    for obj, count in object_counts.items():
        if count == 1:
            descriptions.append(f"one {obj}")
        else:
            descriptions.append(f"{count} {obj}s")
    
    if descriptions:
        return "I detect " + ", ".join(descriptions) + " around you."
    else:
        return "The area appears clear."

# File uploader
uploaded_file = st.file_uploader(" Upload a photo", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    image_np = np.array(image)
    
    with st.spinner(" Analyzing scene..."):
        detections = detect_objects(image_np)
        description = create_description(detections)
    
    st.success(f"** ASSISTANCE:** {description}")
    
    try:
        tts = gTTS(text=description, lang='en', slow=False)
        tts.save("assistance.mp3")
        st.audio("assistance.mp3")
    except:
        st.write(" " + description)
