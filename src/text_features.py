from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf(texts):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5
    )
    X = vectorizer.fit_transform(texts)
    return X
