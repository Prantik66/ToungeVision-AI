import streamlit as st
from PIL import Image
import tempfile
import os
import sys

# allow import from parent folder
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from inference import predict_tongue


st.set_page_config(
    page_title="AI Tongue Diabetes Diagnosis",
    layout="centered"
)

st.title("🩺 AI Tongue Diabetes Diagnosis")
st.write("Upload a tongue image for prediction.")


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):
        with st.spinner("Analyzing..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            result, confidence = predict_tongue(temp_path)

            os.remove(temp_path)

        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}%")