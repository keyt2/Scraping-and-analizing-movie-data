import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import re
import ast
import unicodedata
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def clean_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c)) # deleting diacritics
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def sentiment_analysis(text_series):
    model_name = "tabularisai/multilingual-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)
    results = sentiment_model(text_series.tolist(), batch_size=32, truncation=True)
    sentiments = [r['label'] for r in results]
    return sentiments

def main():
    films = pd.read_csv("filmweb_films.csv")
    films = films[films['rating'] != 0]
    films['description_clean'] = films['description'].apply(clean_text)

    # transforming strings with film genres to lists
    films['genres'] = films['genres'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()

    # encoding genres for ml model
    genre_encoded = mlb.fit_transform(films['genres'])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
    films = pd.concat([films.reset_index(drop=True), genre_df], axis=1)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # encoding descriptions
    descriptions_encoded = model.encode(films['description_clean'].reset_index(drop=True))
    embedding_df = pd.DataFrame(descriptions_encoded, columns=[f"desc_emb_{i}" for i in range(descriptions_encoded.shape[1])])
    films = pd.concat([films.reset_index(drop=True), embedding_df], axis=1)

    films['sentiment'] = sentiment_analysis(films['description_clean'])

    films.to_csv("films_with_emb_+_sent")

if __name__ == "__main__":
    main()
