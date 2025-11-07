---
marp: true
theme: gaia
paginate: true
headingDivider: 2
---

# 🎵 Music Genre Classification  
### GTZAN Dataset Analysis  
🎓 *IDSE Project* by:
  - Shanmukh (CS23B1003)
  - Sohan Kumar (CS23B1004)
  - Kolla Revanth (CS23B1085)

---

## 1️⃣ Problem Statement & Objective
- Automatic classification of songs into **10 genres**
- Applications:
  - 🎧 Music recommendations  
  - 📚 Content organization  
  - 🎯 Streaming platform optimization  

**Goal:**  
Build an ML model to classify tracks using extracted audio features.  

**Genres:** Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock

---

## 2️⃣ Dataset Overview
- **Dataset:** GTZAN Music Genre Collection  
- **Tracks:** 1000 (10 genres × 100 songs)  
- **Duration:** 30 seconds each  
- **Features Extracted:**
  - MFCCs (20 coefficients)
  - Spectral (centroid, bandwidth, rolloff)
  - RMS Energy (loudness)
  - Chroma, Tempo, Harmony

---

## 3️⃣ Data Preprocessing
- Handle missing values & duplicates  
- Genre-wise outlier capping (**3×IQR**)  
- Feature scaling (**StandardScaler**)  
- Dimensionality reduction:
  - **PCA → 39 components (95% variance)**
  - **LDA → 9 components (C–1 rule)**

> 🎵 *3×IQR retains valid genre variations like Metal’s loudness while removing anomalies.*

---

## 4️⃣ Exploratory Data Analysis (EDA)
**Analyses Performed:**
- 📊 Univariate → Feature distributions  
- 🎭 Bivariate → Boxplots (Feature vs Genre)  
- 🧩 Multivariate → Pairplots  
- 🔗 Correlation → Heatmap  
- 🎧 Genre similarity → Cosine matrix  

**Insights:**  
- Metal & Rock → higher loudness & brightness  
- Classical & Jazz → slower tempo, softer tones  
- Pop & Disco → clustered around 120–140 BPM

---

## 5️⃣ Hypothesis Testing
| Test | Feature | Finding |
|------|----------|----------|
| **t-Test** | RMS (Metal vs Classical) | Metal louder ✅ |
| **ANOVA** | Tempo (All genres) | Tempo differs ✅ |
| **Z-Test** | Spectral Centroid | Rock brighter ✅ |

> Confirms statistically that acoustic features differ significantly across genres.

---

## 6️⃣ Model Training
Trained Models:
- Logistic Regression  
- Random Forest  
- SVM  
- **KNN (Best)**  

| Model | Accuracy (PCA) |
|--------|----------------|
| Logistic | 68.2% |
| Random Forest | 77.5% |
| SVM | 80.6% |
| **KNN** | **84.9%** |

---

## 7️⃣ PCA vs LDA
| Aspect | PCA | LDA |
|--------|------|------|
| Type | Unsupervised | Supervised |
| Components | 39 | 9 (C–1) |
| Captures | Non-linear variance | Linear separation |
| Accuracy | ✅ Higher | Lower |

> **Conclusion:** PCA preserved richer, more expressive patterns → better generalization.

---

## 8️⃣ Deployment (Streamlit App)
🎵 **Pipeline:**  
Audio → Feature Extraction → PCA → KNN → Genre Output  

**App Features:**  
- Upload `.wav` file  
- View waveform + spectrogram  
- Display top 3 predicted genres with probabilities  
- View extracted feature values  

> Deployed using **Streamlit** for real-time inference.

---

## 9️⃣ Results Summary
| Metric | Score |
|--------|-------|
| **Accuracy** | **84.93%** |
| **Precision** | 85.28% |
| **Recall** | 84.93% |
| **F1-Score** | 84.90% |

🏆 **Best Model:** KNN + PCA  
✅ Robust, non-parametric, works well with scaled, denoised data.

---

## 🔟 Key Insights & Takeaways
- PCA outperformed LDA (richer variance capture)  
- 3×IQR improved model robustness  
- Tempo, loudness, brightness differentiate genres  
- Real-time Streamlit deployment successful  

🎓 *Turning Sound into Science*
