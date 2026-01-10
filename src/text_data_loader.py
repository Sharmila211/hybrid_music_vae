import pandas as pd

def load_lyrics():
    # Load CSVs
    bangla = pd.read_csv("data/raw/bangla_lyrics.csv")
    english = pd.read_csv("data/raw/english_lyrics.csv")

    # Take first column as lyrics text
    bangla_text = bangla.iloc[:, 0].astype(str)
    english_text = english.iloc[:, 0].astype(str)

    # ---- ADD THESE 3 LINES (IMPORTANT) ----
    bangla_text = bangla_text.dropna()
    english_text = english_text.dropna()
    bangla_text = bangla_text.str.lower()
    english_text = english_text.str.lower()
    # --------------------------------------

    texts = list(bangla_text) + list(english_text)
    labels = (["bangla"] * len(bangla_text)) + (["english"] * len(english_text))

    return texts, labels


if __name__ == "__main__":
    texts, labels = load_lyrics()
    print("Total samples:", len(texts))
    print("Sample text:", texts[0][:100])
