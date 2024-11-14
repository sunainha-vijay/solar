

import streamlit as st
from PIL import Image
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import base64
import io
# Load the pre-trained model
model = load_model('LSTM + GRU 2016 with recurrent_dropout.h5')

# Set page configuration
st.set_page_config(
    page_title="☀️ Solar Flare Prediction ☀️",
    page_icon=":sun:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load the sun image
sun_image = Image.open("sun.png")

# Convert the image to a base64 string
buffered = io.BytesIO()
sun_image.save(buffered, format="PNG")
sun_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

# Add a title and description with the sun image
st.markdown(
    f"""
    <div style="text-align: center;">
        <h1>☀️ Solar Flare Prediction ☀️</h1>
        <img src="data:image/png;base64,{sun_image_base64}" width="200" />
        <p>Use this app to predict the likelihood of solar flares based on your data.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Add some custom CSS to style the app
st.write("""
<style>
    [data-testid="stFileUploader"] button {
        background-color: #0072C6;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        border: none;
        font-size: 1rem;
        cursor: pointer;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #005A9E;
    }
    [data-testid="stExpander"] summary {
        background-color: #F0F0F0;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
        cursor: pointer;
    }
    [data-testid="stExpander"][open] summary {
        background-color: #E0E0E0;
    }
    [data-testid="stContainer"] {
        background-color: #F8F8F8;
        padding: 2rem;
        border-radius: 0.5rem;
        box-shadow: 0px 0px 20px 0px rgba(0,0,0,0.1);
    }
    [data-testid="stMarkdownContainer"] {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# File upload section
with st.expander("Upload Solar Flare Data"):
    uploaded_file = st.file_uploader("Choose a CSV file with solar flare data", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    solar_flare = pd.read_csv(uploaded_file)

    # Preprocess the data
    sequence_length = 5
    num_features = 4
    input_data = solar_flare[['duration.s', 'peak.c/s', 'total.counts', 'radial']].values
    input_data = input_data[:sequence_length].reshape((1, sequence_length, num_features))

    # Make a prediction
    prediction = model.predict(input_data)

    # Interpret the prediction
    if prediction[0][0] >= 0.5:
        st.markdown("## Prediction: Solar flare likely ☀️")
    else:
        st.markdown("## Prediction: No solar flare 🌕")

    # Assuming multi-class classification
    classes = ['A-class', 'B-class', 'C-class', 'M-class', 'X-class']
    predicted_class = np.argmax(prediction)
    st.markdown(f"## Predicted class: {classes[predicted_class]}")
