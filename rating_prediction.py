import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as rmse


def main():
    films = pd.read_csv("films_with_emb_+_sent")

    # separating columns with embedded descriptions
    embedding_cols = [col for col in films.columns if col.startswith('desc_emb_')]

    # separating columns with encoded genres
    non_feature_cols = ['id', 'title', 'year', 'rating', 'votes', 'description', 'genres', 'description_clean', 'sentiment']
    genre_cols = [col for col in films.columns if col not in embedding_cols + non_feature_cols]
    # sentiment_dummies = pd.get_dummies(films['sentiment'], prefix='sent')

    # combining genre and descriptions features
    X = pd.concat([films[embedding_cols + genre_cols]], axis=1)

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
