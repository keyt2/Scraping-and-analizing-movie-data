import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    films = pd.read_csv('films_with_emb_+_sent')
    # correlation between description sentiment and rating
    sentiment_dummies = pd.get_dummies(films['sentiment'], prefix='sent')

    # no significant correlation
    print(films[['rating']].join(sentiment_dummies).corr()['rating'])

    # correlation between genre and rating
    embedding_cols = [col for col in films.columns if col.startswith('desc_emb_')]
    non_feature_cols = ['id', 'title', 'year', 'rating', 'votes', 'description', 'genres', 'description_clean',
                        'sentiment']
    genre_cols = [col for col in films.columns if col not in embedding_cols + non_feature_cols]
    genre_ratings = {
        genre: films.loc[films[genre] == 1, 'rating'].mean()
        for genre in genre_cols
    }
    genre_ratings_df = pd.Series(genre_ratings).sort_values(ascending=False)

    # visualization
    plt.figure(figsize=(15, 8))
    sns.barplot(x=genre_ratings_df.values, y=genre_ratings_df.index)
    plt.xlabel('Average rating')
    plt.ylabel('Genre')
    plt.title('Average rating by genre')
    plt.show()

if __name__ == "__main__":
    main()