"""
Dashboard interativo do projeto Spotify Genre Analytics.
Rodar com: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_connection, MODELS_DIR

st.set_page_config(page_title="Spotify Genre Analytics", page_icon="🎵", layout="wide")

NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]
CATEGORICAL_FEATURES = ["key", "mode", "time_signature", "explicit"]
FIGURES_DIR = ROOT / "outputs" / "figures"

# Tradução dos nomes técnicos pra exibição
FEATURE_LABELS = {
    "danceability": "Dançabilidade",
    "energy": "Energia",
    "loudness": "Volume (dB)",
    "speechiness": "Presença de Fala",
    "acousticness": "Acusticidade",
    "instrumentalness": "Instrumentalidade",
    "liveness": "Ao Vivo",
    "valence": "Positividade (Valence)",
    "tempo": "Andamento (BPM)",
}


@st.cache_data
def load_data():
    con = get_connection()
    df = con.execute("SELECT * FROM tracks_clean WHERE genre_macro != 'other'").fetchdf()
    con.close()
    return df


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODELS_DIR / "genre_classifier.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    labels = joblib.load(MODELS_DIR / "labels.pkl")
    return model, scaler, labels


df = load_data()
model, scaler, labels = load_model_artifacts()

st.title("🎵 Spotify Genre Analytics")
st.caption("Análise exploratória + classificador de gênero musical a partir de características de áudio (114 mil faixas, Kaggle)")

# ============================================================
# BARRA LATERAL - FILTROS
# ============================================================
st.sidebar.header("🎛️ Filtros")

genres_all = sorted(df["genre_macro"].unique())
selected_genres = st.sidebar.multiselect("Gênero", genres_all, default=genres_all)

pop_range = st.sidebar.slider("Popularidade", 0, 100, (0, 100))

dur_min, dur_max = float(df["duration_min"].min()), float(df["duration_min"].max())
duration_range = st.sidebar.slider("Duração (min)", dur_min, dur_max, (dur_min, dur_max))

explicit_filter = st.sidebar.radio(
    "Conteúdo explícito", ["Todas", "Somente explícitas", "Somente não-explícitas"]
)

filtered = df[
    df["genre_macro"].isin(selected_genres)
    & df["popularity"].between(*pop_range)
    & df["duration_min"].between(*duration_range)
]
if explicit_filter == "Somente explícitas":
    filtered = filtered[filtered["explicit"]]
elif explicit_filter == "Somente não-explícitas":
    filtered = filtered[~filtered["explicit"]]

st.sidebar.markdown("---")
st.sidebar.caption(f"**{len(filtered):,}** faixas no filtro atual (de {len(df):,} no total)")

if len(filtered) == 0:
    st.warning("Nenhuma faixa corresponde aos filtros selecionados. Ajuste os filtros na barra lateral.")
    st.stop()

# ============================================================
# ESTATÍSTICAS GERAIS
# ============================================================
st.subheader("📌 Estatísticas gerais")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Faixas filtradas", f"{len(filtered):,}")
k2.metric("Gêneros", filtered["genre_macro"].nunique())
k3.metric("Popularidade média", f"{filtered['popularity'].mean():.1f}")
k4.metric("Duração média", f"{filtered['duration_min'].mean():.1f} min")
k5.metric("% faixas explícitas", f"{filtered['explicit'].mean() * 100:.1f}%")

st.markdown("---")


def bar_chart(data, title, xlabel, color_palette="viridis", ascending=True):
    """Gráfico de barras horizontal simples, ordenado."""
    data = data.sort_values(ascending=ascending)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = sns.color_palette(color_palette, len(data))
    ax.barh(data.index.astype(str), data.values, color=colors)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ============================================================
# TELA INICIAL - 5 GRÁFICOS + INSIGHTS
# ============================================================
st.header("📊 Visão Geral")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1. Quantidade de faixas por gênero")
    contagem = filtered["genre_macro"].value_counts()
    st.pyplot(bar_chart(contagem, "Faixas por gênero", "Quantidade de faixas"))
    genero_maior = contagem.idxmax()
    st.info(f"💡 **{genero_maior}** é o gênero com mais faixas no filtro atual ({contagem.max():,}).")

with col2:
    st.markdown("#### 2. Popularidade média por gênero")
    popularidade = filtered.groupby("genre_macro")["popularity"].mean().round(1)
    st.pyplot(bar_chart(popularidade, "Popularidade média por gênero", "Popularidade"))
    genero_popular = popularidade.idxmax()
    st.info(f"💡 **{genero_popular}** tem a maior popularidade média ({popularidade.max():.1f}), mesmo sem ser o gênero com mais faixas.")

col3, col4 = st.columns(2)

with col3:
    st.markdown("#### 3. Energia mediana por gênero")
    energia = filtered.groupby("genre_macro")["energy"].median().round(3)
    st.pyplot(bar_chart(energia, "Energia mediana por gênero", "Energia", color_palette="rocket"))
    genero_energia = energia.idxmax()
    st.info(f"💡 **{genero_energia}** é disparado o gênero mais energético do dataset.")

with col4:
    st.markdown("#### 4. Dançabilidade mediana por gênero")
    danca = filtered.groupby("genre_macro")["danceability"].median().round(3)
    st.pyplot(bar_chart(danca, "Dançabilidade mediana por gênero", "Dançabilidade", color_palette="crest"))
    genero_danca = danca.idxmax()
    st.info(f"💡 **{genero_danca}** lidera em dançabilidade — bom indício de faixas feitas pra pista.")

st.markdown("#### 5. Positividade (valence) mediana por gênero")
valencia = filtered.groupby("genre_macro")["valence"].median().round(3)
st.pyplot(bar_chart(valencia, "Positividade mediana por gênero", "Valence", color_palette="mako"))
genero_valencia = valencia.idxmax()
st.info(f"💡 **{genero_valencia}** tem o som mais 'positivo' do dataset — faixas soam mais alegres/animadas.")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["🔍 Exploração Detalhada", "🔮 Preditor de Gênero", "🎯 Desempenho do Modelo"]
)

# ============================================================
# TAB 1: EXPLORAÇÃO DETALHADA
# ============================================================
with tab1:
    st.header("Exploração detalhada")

    st.subheader("Correlação entre características de áudio")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    corr = filtered[NUMERIC_FEATURES + ["popularity"]].rename(columns=FEATURE_LABELS).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader("Comparar duas características (dispersão)")
    st.caption("Visualização opcional — escolha as características pra comparar visualmente.")
    colx, coly = st.columns(2)
    with colx:
        feature_x = st.selectbox(
            "Característica (eixo X)", NUMERIC_FEATURES,
            format_func=lambda x: FEATURE_LABELS[x], index=1,
        )
    with coly:
        feature_y = st.selectbox(
            "Característica (eixo Y)", NUMERIC_FEATURES,
            format_func=lambda x: FEATURE_LABELS[x], index=0,
        )

    sample = filtered.sample(min(3000, len(filtered)), random_state=42)
    fig_scatter, ax_scatter = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=sample, x=feature_x, y=feature_y, hue="genre_macro",
        palette="viridis", alpha=0.5, s=18, ax=ax_scatter,
    )
    ax_scatter.set_xlabel(FEATURE_LABELS[feature_x])
    ax_scatter.set_ylabel(FEATURE_LABELS[feature_y])
    ax_scatter.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Gênero")
    st.pyplot(fig_scatter)

    st.subheader("Top 15 faixas mais populares (no filtro)")
    top_tracks = (
        filtered.sort_values("popularity", ascending=False)
        .head(15)[["track_name", "artists", "genre_macro", "popularity"]]
        .rename(columns={
            "track_name": "Faixa", "artists": "Artista(s)",
            "genre_macro": "Gênero", "popularity": "Popularidade",
        })
        .reset_index(drop=True)
    )
    st.dataframe(top_tracks, use_container_width=True)

    st.subheader("Estatísticas medianas por gênero")
    stats_table = filtered.groupby("genre_macro")[NUMERIC_FEATURES + ["popularity"]].median().round(3)
    stats_table = stats_table.rename(columns=FEATURE_LABELS)
    st.dataframe(stats_table, use_container_width=True)

# ============================================================
# TAB 2: PREDITOR DE GÊNERO
# ============================================================
with tab2:
    st.header("Preditor de gênero")
    st.write("Ajuste os controles pra simular as características sonoras de uma faixa e veja qual gênero o modelo prevê.")

    col1, col2, col3 = st.columns(3)

    with col1:
        danceability = st.slider("Dançabilidade", 0.0, 1.0, 0.5, 0.01)
        energy = st.slider("Energia", 0.0, 1.0, 0.5, 0.01)
        loudness = st.slider("Volume (dB)", -60.0, 0.0, -10.0, 0.5)
        speechiness = st.slider("Presença de Fala", 0.0, 1.0, 0.1, 0.01)

    with col2:
        acousticness = st.slider("Acusticidade", 0.0, 1.0, 0.3, 0.01)
        instrumentalness = st.slider("Instrumentalidade", 0.0, 1.0, 0.0, 0.01)
        liveness = st.slider("Ao Vivo", 0.0, 1.0, 0.15, 0.01)
        valence = st.slider("Positividade (Valence)", 0.0, 1.0, 0.5, 0.01)

    with col3:
        tempo = st.slider("Andamento (BPM)", 40, 220, 120, 1)
        key = st.selectbox("Tom", list(range(-1, 12)), index=1)
        mode = st.selectbox("Modo (0 = Menor, 1 = Maior)", [0, 1], index=1)
        time_signature = st.selectbox("Compasso", [3, 4, 5, 6, 7], index=1)
        explicit = st.checkbox("Conteúdo Explícito", value=False)

    input_df = pd.DataFrame([{
        "danceability": danceability, "energy": energy, "loudness": loudness,
        "speechiness": speechiness, "acousticness": acousticness,
        "instrumentalness": instrumentalness, "liveness": liveness,
        "valence": valence, "tempo": tempo, "key": key, "mode": mode,
        "time_signature": time_signature, "explicit": int(explicit),
    }])

    input_scaled = input_df.copy()
    input_scaled[NUMERIC_FEATURES] = scaler.transform(input_df[NUMERIC_FEATURES])

    proba = model.predict_proba(input_scaled[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0]
    pred_genre = model.classes_[np.argmax(proba)]
    confidence = np.max(proba)

    st.subheader(f"Previsão: **{pred_genre}**")
    st.caption(f"Confiança do modelo: {confidence * 100:.1f}%")

    proba_df = pd.DataFrame({"Gênero": model.classes_, "Probabilidade": proba}).sort_values(
        "Probabilidade", ascending=True
    )
    fig_proba, ax_proba = plt.subplots(figsize=(8, 5))
    ax_proba.barh(proba_df["Gênero"], proba_df["Probabilidade"], color="mediumseagreen")
    ax_proba.set_xlabel("Probabilidade")
    ax_proba.set_title("Probabilidade por gênero")
    st.pyplot(fig_proba)

# ============================================================
# TAB 3: DESEMPENHO DO MODELO
# ============================================================
with tab3:
    st.header("Desempenho do modelo")

    st.info(
        "O modelo foi treinado **sem** a variável de popularidade — só com características de áudio. "
        "Isso é proposital: o objetivo é testar se o gênero é reconhecível pelo som puro, "
        "não pelo quanto a faixa é tocada no mercado."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(str(FIGURES_DIR / "confusion_matrix_random_forest.png"), caption="Matriz de confusão - Random Forest")
    with col2:
        st.image(str(FIGURES_DIR / "feature_importance.png"), caption="Importância das características")

    st.subheader("🎲 Veja o modelo em ação")
    st.caption("Sorteia faixas do filtro atual e compara o gênero real com a previsão do modelo.")

    if st.button("Sortear 10 faixas"):
        sample_preds = filtered.sample(min(10, len(filtered)), random_state=np.random.randint(0, 10000))
        X_sample = sample_preds[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
        X_sample["explicit"] = X_sample["explicit"].astype(int)
        X_sample[NUMERIC_FEATURES] = scaler.transform(X_sample[NUMERIC_FEATURES])

        preds = model.predict(X_sample)
        result = pd.DataFrame({
            "Faixa": sample_preds["track_name"].values,
            "Artista(s)": sample_preds["artists"].values,
            "Gênero real": sample_preds["genre_macro"].values,
            "Previsto": preds,
        })
        result["Acertou"] = result["Gênero real"] == result["Previsto"]
        result["Acertou"] = result["Acertou"].map({True: "✅", False: "❌"})
        st.dataframe(result, use_container_width=True)
        st.caption(f"Acertos nessa amostra: {(result['Acertou'] == '✅').sum()}/10")

    st.subheader("Principais achados sobre o modelo")
    st.markdown("""
- O Random Forest atingiu **~40% de acurácia** classificando gênero **só pelo áudio**, sem usar popularidade.
- Esse resultado é **honesto, não uma falha**: gênero musical é em parte uma categoria cultural e de marketing, não puramente acústica — existe um limite teórico pra esse tipo de classificação.
- **Metal e classical são os gêneros mais "reconhecíveis"** pelo som puro.
- **Pop e rock são os mais ambíguos**, porque funcionam como "guarda-chuvas" que se sobrepõem sonoramente com quase todos os outros gêneros.
- **Acusticidade foi a característica mais importante** do modelo, seguida de dançabilidade e energia.
    """)