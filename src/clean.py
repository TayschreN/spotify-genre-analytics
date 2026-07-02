"""
Limpeza dos dados: remove duplicatas, trata nulos, mapeia gêneros,
remove outliers grosseiros e salva tabela limpa em parquet + DuckDB.
"""
import pandas as pd
from utils import get_connection, DATA_PROCESSED, map_genre


def clean():
    con = get_connection()
    df = con.execute("SELECT * FROM raw_tracks").fetchdf()
    print(f"Linhas brutas: {len(df)}")

    # 1. Remove duplicatas — mesma música aparece em múltiplos álbuns/playlists
    before = len(df)
    df = df.drop_duplicates(subset=["artists", "track_name"], keep="first")
    print(f"Removidas {before - len(df)} duplicatas (artists + track_name)")

    # 2. Remove linhas com campos essenciais nulos
    essential = ["track_id", "artists", "track_name", "track_genre"]
    before = len(df)
    df = df.dropna(subset=essential)
    print(f"Removidas {before - len(df)} linhas com campos essenciais nulos")

    # 3. Remove a coluna de índice fantasma do Kaggle, se existir
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # 4. Trata outliers grosseiros de duration_ms
    # faixas < 30s ou > 20min provavelmente são lixo/interlúdio/erro
    before = len(df)
    df = df[(df["duration_ms"] >= 30_000) & (df["duration_ms"] <= 1_200_000)]
    print(f"Removidas {before - len(df)} faixas com duração inválida (<30s ou >20min)")

    # 5. Garante tipos corretos
    df["explicit"] = df["explicit"].astype(bool)
    df["popularity"] = df["popularity"].astype(int)

    # 6. Cria macro-gênero
    df["genre_macro"] = df["track_genre"].apply(map_genre)
    print("\nDistribuição de macro-gêneros:")
    print(df["genre_macro"].value_counts())

    # 7. Cria coluna de duração em minutos (mais legível pra EDA/dashboard)
    df["duration_min"] = (df["duration_ms"] / 60_000).round(2)

    print(f"\nLinhas finais: {len(df)}")

    # Salva em parquet (pra reuso rápido) e registra como tabela no DuckDB
    out_path = DATA_PROCESSED / "tracks_clean.parquet"
    df.to_parquet(out_path, index=False)

    con.execute("DROP TABLE IF EXISTS tracks_clean")
    con.execute(f"CREATE TABLE tracks_clean AS SELECT * FROM read_parquet('{out_path}')")
    print(f"\n✅ Salvo em {out_path} e tabela tracks_clean no DuckDB")

    con.close()
    return df


if __name__ == "__main__":
    clean()