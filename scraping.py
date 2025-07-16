import requests
import csv
import time
from bs4 import BeautifulSoup

BASE_URL = "https://www.filmweb.pl/api/v1/film/"
POPULAR_URL = "https://www.filmweb.pl/api/v1/film/popular?all=true"
FILM_URL = "https://www.filmweb.pl/film/"

headers = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0"
}

def get_popular_film_ids():
    response = requests.get(POPULAR_URL, headers=headers)
    response.raise_for_status()
    return [film for film in response.json()]

# we can scrape film genre separately from HTML only by using
# info about title, year and id
def get_film_genre(title, year, id):
    film_url = f"{FILM_URL}{title}-{year}-{id}"
    response = requests.get(film_url, headers=headers)
    if response.status_code == 200 :
        soup = BeautifulSoup(response.text, "html.parser")
        genre_spans = soup.find_all('span', itemprop='genre')
        genres = [span.text.strip() for span in genre_spans]
        return genres

def get_film_data(film_id):
    try:
        rating = requests.get(f"{BASE_URL}{film_id}/rating", headers=headers).json()
        description = requests.get(f"{BASE_URL}{film_id}/description", headers=headers).json()
        info = requests.get(f"{BASE_URL}{film_id}/info", headers=headers).json()

        title = info.get("title")
        year = info.get("year")
        genres = get_film_genre(title, year, film_id)

        return {
            "id": film_id,
            "title": info.get("title"),
            "year": info.get("year"),
            "rating": rating.get("rate"),
            "votes": rating.get("count"),
            "description": description.get("synopsis"),
            "genres": genres
        }
    except Exception as e:
        print(f"Error for film {film_id}: {e}")
        return None

def main():
    film_ids = get_popular_film_ids()
    print(f"{len(film_ids)} film IDs were downloaded.")

    with open("filmweb_films.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["id", "title", "year", "rating", "votes", "description", "genres"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, film_id in enumerate(film_ids):
            film_data = get_film_data(film_id)
            if film_data:
                writer.writerow(film_data)
                print(f"[{i+1}/{len(film_ids)}]")
            time.sleep(0.5)  # small delay to avoid being blocked

if __name__ == "__main__":
    main()
