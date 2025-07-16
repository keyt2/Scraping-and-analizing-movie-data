import pandas as pd
import numpy as np
import re
import unicodedata
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as rmse


def clean_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c)) # deleting diacritics
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    films = pd.read_csv("filmweb_films.csv")
    films = films[films['rating'] != 0]
    films['description_clean'] = films['description'].apply(clean_text)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X = model.encode(films['description_clean'].reset_index(drop=True))

    X_train, X_test, y_train, y_test = train_test_split(X, films[['rating', 'title']], test_size=0.3, random_state=100)
    model = Ridge()
    model.fit(X_train, y_train['rating'])
    y_pred = model.predict(X_test)
    rmse1 = round(rmse(y_test['rating'], y_pred), 5)
    print(rmse1)

    results = y_test.copy()
    results['true_rating'] = y_test['rating']
    results['predicted_rating'] = y_pred
    results['abs_error'] = np.abs(results['true_rating'] - results['predicted_rating'])
    print(results.sort_values('abs_error', ascending=False)[['title', 'true_rating', 'predicted_rating']].head(10))


if __name__ == "__main__":
    main()
