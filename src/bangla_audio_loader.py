import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_bangla_audio_features():
    df = pd.read_csv("data/raw/bangla_audio/dataset.csv")

    # Drop non-numeric columns
    X = df.select_dtypes(include=["float64", "int64"])

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


if __name__ == "__main__":
    X = load_bangla_audio_features()
    print("Bangla audio feature shape:", X.shape)
