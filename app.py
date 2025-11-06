import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import os
from feature_extractor import GTZANFeatureExtractor  # <-- your custom class

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="🎵 GTZAN Genre Classifier", layout="wide")

st.title("🎧 GTZAN Music Genre Classification")
st.markdown("""# Music Genre Classification using GTZAN Dataset

## 1. Business Understanding

### Problem Statement
Music genre classification is a fundamental problem in the field of music information retrieval (MIR). With the exponential growth of digital music libraries and streaming platforms, automatically categorizing music into genres has become crucial for:

- **Music Discovery**: Helping users find new music based on their preferences
- **Content Organization**: Organizing large digital music libraries efficiently
- **Recommendation Systems**: Powering music recommendation algorithms
- **Intellectual Property**: Assisting in copyright detection and management

### Project Objective
The goal of this project is to build a machine learning model that can accurately classify audio tracks into one of 10 distinct music genres using extracted audio features from the GTZAN dataset.

**Genres to Classify**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock

### Success Criteria
- Achieve high classification accuracy across all genres
- Build a robust model that generalizes well to unseen audio data
- Understand which audio features are most discriminative for genre classification

This analysis will follow a comprehensive ML pipeline: data collection, preprocessing, exploratory analysis, feature engineering, model training, and evaluation.
""")
st.write("")  # 1 line space
st.write("")  # 2 line space
st.markdown("Upload a **.wav** file to visualize its waveform, spectrogram, predicted genre, and probabilities.")

# ---------------------------
# Load pipeline and mapping
# ---------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load(r'Pipelines/final_pipeline.pkl')

@st.cache_data
def load_mapping():
    mapping = pd.read_csv(r'cleaned_data/label_mapping.csv')
    return dict(zip(mapping['Encoded_Value'], mapping['Genre']))

genre_pipeline = load_pipeline()
genre_mapping = load_mapping()

# ---------------------------
# File upload
# ---------------------------
uploaded_file = st.file_uploader("🎵 Upload a music file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save temporarily for librosa
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # ---------------------------
    # Audio Visualization
    # ---------------------------
    y, sr = librosa.load(temp_path, duration=30)

    st.subheader("🎶 Audio Waveform and Spectrogram")

    col1, col2 = st.columns(2)

    # Waveform
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='steelblue')
        ax.set_title("Waveform", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    # Spectrogram
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
        ax.set_title("Mel Spectrogram", fontsize=12, fontweight='bold')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        st.pyplot(fig)

    # ---------------------------
    # Prediction Section
    # ---------------------------
    try:
        pred = genre_pipeline.predict([temp_path])[0]
        predicted_genre = genre_mapping.get(pred, str(pred))
        probs = genre_pipeline.predict_proba([temp_path])[0]
        proba_dict = {genre_mapping[i]: float(p) for i, p in enumerate(probs)}
        top3 = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.stop()

    # Display prediction
    st.subheader("🎯 Predicted Genre:")
    st.success(f"**{predicted_genre}**")

    # Probability Bar Chart
    st.subheader("📊 Genre Confidence Levels")
    top3_df = pd.DataFrame(top3, columns=["Genre", "Probability"])
    top3_df = top3_df.set_index("Genre")
    st.bar_chart(top3_df)

    # ---------------------------
    # Feature Extraction Display
    # ---------------------------
    st.subheader("📋 Extracted Audio Features (58 values)")

    try:
        feature_extractor = genre_pipeline.named_steps['feature_extractor']
        features = feature_extractor.extract_gtzan_features(temp_path)
        features_df = pd.DataFrame([features]).T.reset_index()
        features_df.columns = ['Feature', 'Value']

        st.dataframe(features_df, use_container_width=True)

        # CSV download option
        csv = features_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Features as CSV",
            data=csv,
            file_name="audio_features.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.warning(f"⚠️ Could not extract features: {e}")

else:
    st.info("Please upload a .wav file to begin 🎵")
