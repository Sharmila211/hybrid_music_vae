import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_audio_features(csv_path):
    df = pd.read_csv(csv_path)

    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c.startswith("mfcc")]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X
