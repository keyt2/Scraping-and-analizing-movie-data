Scraping an anlysing data about movies from filmweb.pl website.



scraping.py scrapes data and saves it in filmweb_films_csv

preprocessing.py preprocesses data, encodes genre, adds embeddings of descritions and description sentiment, then it saves data to films_with_emb_+_sent

rating_prediciton.py uses preprocessed data to predict rating with linear regression based on movie descriptions

correlation.py examines correlation between description and rating and movie genre and rating
