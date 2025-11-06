import librosa
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # optional: avoid CPU core warning

class GTZANFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, duration=30, columns_order=None):
        self.duration = duration
        self.columns_order = columns_order  # for consistent column order

    def extract_gtzan_features(self, audio_path):
        y, sr = librosa.load(audio_path, duration=self.duration)
        features = {}

        features['length'] = len(y)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = np.mean(chroma_stft)
        features['chroma_stft_var'] = np.var(chroma_stft)

        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)

        features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_centroid_var'] = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['spectral_bandwidth_var'] = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features['rolloff_var'] = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))

        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_var'] = np.var(zcr)

        harmony = librosa.effects.harmonic(y)
        perceptr = librosa.effects.percussive(y)
        features['harmony_mean'] = np.mean(harmony)
        features['harmony_var'] = np.var(harmony)
        features['perceptr_mean'] = np.mean(perceptr)
        features['perceptr_var'] = np.var(perceptr)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            features[f'mfcc{i}_mean'] = np.mean(mfccs[i-1])
            features[f'mfcc{i}_var'] = np.var(mfccs[i-1])

        return features

    def transform(self, X, y=None):
        feature_list = [self.extract_gtzan_features(x) for x in X]
        df = pd.DataFrame(feature_list)

        # Align columns to match training pipeline
        if self.columns_order is not None:
            df = df.reindex(columns=self.columns_order, fill_value=0)
        return df

    def fit(self, X, y=None):
        return self