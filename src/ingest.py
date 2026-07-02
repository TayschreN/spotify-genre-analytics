"""
Carrega o CSV bruto do Spotify para uma tabela no DuckDB.
Baixe o dataset em: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
e coloque o CSV em data/raw/spotify_tracks.csv
"""
from utils import get_connection, DATA_RAW


def ingest():
    csv_path = DATA_RAW / "spotify_tracks.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV não encontrado em {csv_path}. Baixe o dataset do Kaggle e coloque aqui."
        )

    con = get_connection()

    con.execute("DROP TABLE IF EXISTS raw_tracks")
    con.execute(f"""
        CREATE TABLE raw_tracks AS
        SELECT * FROM read_csv_auto('{csv_path}', header=True, sample_size=-1)
    """)

    count = con.execute("SELECT COUNT(*) FROM raw_tracks").fetchone()[0]
    print(f"✅ Ingeridos {count} registros em raw_tracks")

    print("\nSchema:")
    print(con.execute("DESCRIBE raw_tracks").fetchdf())

    con.close()


if __name__ == "__main__":
    ingest()