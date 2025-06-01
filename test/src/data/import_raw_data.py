import requests
import os
import logging


def import_raw_data(raw_data_relative_path,
                    filenames,
                    bucket_folder_url):
    """Import filenames from bucket_folder_url into raw_data_relative_path."""
    os.makedirs(raw_data_relative_path, exist_ok=True)

    for filename in filenames:
        output_file = os.path.join(raw_data_relative_path, filename)

        if os.path.exists(output_file):
            print(f"✅ Datei bereits vorhanden, überspringe: {filename}")
            continue

        url = os.path.join(bucket_folder_url, filename)
        print(f"⬇️  Lade herunter: {url}")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            with open(output_file, "wb") as f:
                f.write(response.content)

            print(f"✅ Gespeichert unter: {output_file}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Fehler beim Download von {url}: {e}")


def main(raw_data_relative_path="./data/raw",
         filenames=None,
         bucket_folder_url="https://mlops-project-db.s3.eu-west-1.amazonaws.com/movie_recommandation/"):
    if filenames is None:
        filenames = [
            "genome-scores.csv", "genome-tags.csv", "links.csv",
            "movies.csv", "ratings.csv", "README.txt", "tags.csv"
        ]

    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    logging.getLogger(__name__).info("✅ Rohdatenprüfung abgeschlossen.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()