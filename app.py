
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ✅ Load the trained model (make sure hematovision_model.h5 is in the same folder)
model = load_model("hematovision_model.h5")

# ✅ Define the blood cell classes
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# ✅ Streamlit UI
st.set_page_config(page_title="HematoVision", layout="centered")
st.title("🧪 HematoVision: Blood Cell Classifier")

st.write("Upload a blood cell image to classify it into one of the 4 categories.")

# File uploader
uploaded_file = st.file_uploader("Upload JPG/PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🧬 Predicted Cell Type: **{predicted_class}**")
