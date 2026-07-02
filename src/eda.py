"""
Gera análises exploratórias e salva gráficos em outputs/figures/.
Pensado pra rodar standalone ou ser chamado célula a célula no notebook.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_connection, ROOT

FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]


def load_data():
    con = get_connection()
    df = con.execute("SELECT * FROM tracks_clean WHERE genre_macro != 'other'").fetchdf()
    con.close()
    return df


def plot_genre_distribution(df):
    plt.figure(figsize=(10, 6))
    order = df["genre_macro"].value_counts().index
    sns.countplot(data=df, y="genre_macro", order=order, hue="genre_macro",
                  palette="viridis", legend=False)
    plt.title("Distribuição de faixas por macro-gênero")
    plt.xlabel("Quantidade de faixas")
    plt.ylabel("Gênero")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "genre_distribution.png", dpi=120)
    plt.close()
    print("✅ genre_distribution.png salvo")


def plot_feature_correlation(df):
    plt.figure(figsize=(10, 8))
    corr = df[AUDIO_FEATURES + ["popularity"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlação entre audio features e popularidade")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_correlation.png", dpi=120)
    plt.close()
    print("✅ feature_correlation.png salvo")


def plot_feature_by_genre(df, feature="energy"):
    """Gráfico de barras horizontais ordenado pela mediana, com IQR como linha de erro."""
    stats = df.groupby("genre_macro")[feature].agg(
        median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)
    ).sort_values("median", ascending=True)

    plt.figure(figsize=(10, 6))
    y_pos = range(len(stats))
    err_low = stats["median"] - stats["q1"]
    err_high = stats["q3"] - stats["median"]

    plt.barh(y_pos, stats["median"], color=sns.color_palette("viridis", len(stats)))
    plt.errorbar(stats["median"], y_pos, xerr=[err_low, err_high],
                 fmt="none", ecolor="black", capsize=4, alpha=0.6)

    plt.yticks(y_pos, stats.index)
    plt.xlabel(feature)
    plt.title(f"Mediana de '{feature}' por macro-gênero (barra = IQR)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{feature}_by_genre.png", dpi=120)
    plt.close()
    print(f"✅ {feature}_by_genre.png salvo")


def plot_popularity_vs_explicit(df):
    """Histograma sobreposto em vez de boxplot."""
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="popularity", hue="explicit", kde=True,
                 stat="density", common_norm=False, palette="Set2", alpha=0.5)
    plt.title("Distribuição de popularidade: faixas explícitas vs não-explícitas")
    plt.xlabel("Popularidade")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "popularity_vs_explicit.png", dpi=120)
    plt.close()
    print("✅ popularity_vs_explicit.png salvo")


def print_key_stats(df):
    print("\n=== Estatísticas-chave ===")
    print(f"Total de faixas (sem 'other'): {len(df)}")
    print(f"Gêneros: {df['genre_macro'].nunique()}")
    print(f"\nPopularidade média geral: {df['popularity'].mean():.1f}")
    print(f"Popularidade média (explicit=True): {df[df['explicit']]['popularity'].mean():.1f}")
    print(f"Popularidade média (explicit=False): {df[~df['explicit']]['popularity'].mean():.1f}")

    print("\nTop 3 gêneros mais 'dançantes' (mediana danceability):")
    print(df.groupby("genre_macro")["danceability"].median().sort_values(ascending=False).head(3))

    print("\nTop 3 gêneros mais energéticos (mediana energy):")
    print(df.groupby("genre_macro")["energy"].median().sort_values(ascending=False).head(3))

    print("\nGênero mais popular (mediana popularity):")
    print(df.groupby("genre_macro")["popularity"].median().sort_values(ascending=False).head(3))


def run_eda():
    df = load_data()
    print_key_stats(df)
    plot_genre_distribution(df)
    plot_feature_correlation(df)
    plot_feature_by_genre(df, "energy")
    plot_feature_by_genre(df, "danceability")
    plot_feature_by_genre(df, "valence")
    plot_popularity_vs_explicit(df)
    print(f"\n✅ Todos os gráficos salvos em {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
