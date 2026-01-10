import pandas as pd

print("Script started")

df = pd.read_csv("data/raw/bangla_audio/dataset.csv")

print("CSV loaded successfully")
print("Shape:", df.shape)
print("First 5 columns:", list(df.columns)[:5])
print("First row:")
print(df.iloc[0])

print("Script finished")
