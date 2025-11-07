# 🎵 Music Genre Classification using GTZAN Dataset

## 📘 Overview
This project builds a **machine learning pipeline** to classify songs into **10 genres** using the **GTZAN dataset**.  
It extracts **audio features** (MFCCs, spectral, chroma, tempo, RMS, etc.), performs **EDA, PCA/LDA**, and trains multiple models —  
with **KNN + PCA** achieving the best performance (≈87% accuracy).

---

## 🎯 Objectives
- Automatically classify audio tracks into 10 genres:
  **Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock**
- Perform **exploratory data analysis**, **dimensionality reduction**, and **model comparison**
- Deploy a **Streamlit app** for real-time genre prediction

---

## 🎶 Dataset Information
**Dataset:** GTZAN Music Genre Dataset  
**Samples:** 1000 tracks (10 genres × 100 songs)  
**Duration:** 30 seconds per track  
**Features Extracted:**
- MFCCs (20 coefficients)
- Spectral features (centroid, bandwidth, rolloff)
- RMS energy (loudness)
- Chroma & Tempo

---

## 🧹 Data Preprocessing
- Handle missing & duplicate values  
- Genre-wise outlier detection (3×IQR rule)  
- Label encoding for genre names  
- Feature scaling (StandardScaler)  
- Dimensionality reduction using **PCA (39 comps)** & **LDA (9 comps)**

> ✅ **3×IQR** chosen for softer outlier capping → retains valid genre variations.

---

## 📊 Exploratory Data Analysis (EDA)
Performed to understand relationships and patterns in audio features:

| Analysis Type | Description |
|----------------|--------------|
| **Univariate** | Histograms of top features |
| **Bivariate** | Boxplots (feature vs genre) |
| **Multivariate** | Pairplots for feature interaction |
| **Correlation** | Heatmap to detect redundant features |
| **Audio-specific** | Mean feature comparison (MFCCs, tempo, spectral) |
| **Genre Similarity** | Cosine similarity heatmap between genres |
| **Hypothesis Testing** | T-test (Metal vs Classical loudness), ANOVA (tempo differences) |

---

## ⚙️ Model Training
Trained multiple classifiers on both **PCA** and **LDA** reduced datasets:

| Model | Accuracy (PCA) | Accuracy (LDA) |
|--------|----------------|----------------|
| Logistic Regression | 68.2% | 67.9% |
| Random Forest | 77.5% | 75.0% |
| SVM | 80.6% | 73.9% |
| **KNN** | **84.9%** | **78.1%** |

🏆 **Best Model:** KNN + PCA  
→ Non-parametric, adapts to local feature structure, performs well with scaled data.

---

## 📈 Model Insights
- **PCA outperformed LDA** due to richer non-linear variance capture  
- **LDA limited** by class-bound assumptions (C–1 = 9 components)
- PCA (39 comps) preserved maximum discriminative information

---

## 🧪 Hypothesis Testing Highlights
| Test | Feature | Purpose |
|------|----------|----------|
| **t-Test** | `rms_mean` | Are Metal songs louder than Classical? |

This test confirms **genre-level acoustic distinctions** are statistically significant.

---

## 🚀 Deployment (Streamlit App)
**Pipeline:**  
🎵 Raw Audio → 🔍 Feature Extraction → 📐 PCA → 🤖 KNN → 🎭 Genre Prediction

Features:
- Upload `.wav` file  
- Displays waveform & spectrogram  
- Shows top 3 genre probabilities  
- Lists extracted feature values  

---

## 🧠 Key Takeaways
1. **KNN + PCA** achieved **84.9% accuracy**
2. PCA captured non-linear variance better than LDA  
3. Outlier capping (3×IQR) improved model stability  
4. Streamlit app enables **real-time genre prediction**

---

## ⚙️ Setup Instructions

### 🔹 Clone the Repository
```bash
git clone https://github.com/Sohanuu66/Music_Genre_Classification_GTZAN_Dataset.git
cd Music_Genre_Classification_GTZAN_Dataset
```

### 🔹 Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate       # On macOS/Linux
.venv\Scripts\activate          # On Windows
```

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔹 Run Streamlit App
```bash
streamlit run app.py
```
Then open the provided local URL in your browser.

---

## 🧰 Tech Stack
- **Python 3.11**
- **Libraries:** Librosa, NumPy, Pandas, Scikit-learn, Streamlit, Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Model:** KNN + PCA pipeline (saved with Joblib)

---

## 📚 References
- [GTZAN Genre Dataset](http://marsyas.info/downloads/datasets.html)
- [Librosa Documentation](https://librosa.org/doc)
- [Scikit-learn User Guide](https://scikit-learn.org/)
