# 🎵 Spotify Genre Analytics

Pipeline completo de dados — da ingestão à predição — para explorar se **características sonoras** de uma faixa (dançabilidade, energia, acusticidade, etc.) conseguem prever seu **gênero musical**, sem depender de métricas de popularidade ou de marketing.

**[Dashboard interativo (Streamlit)](#-como-rodar)** · **[Notebook de EDA](notebooks/01_eda.ipynb)** · Dataset: [Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)


![Dashboard](imagem/tela1.png)


---

## 📌 Sobre o projeto

Esse projeto nasceu de uma pergunta simples: **dá pra reconhecer o gênero de uma música só pelo jeito que ela soa?** Sem saber o artista, o nome da faixa ou o quão popular ela é — só pelos números brutos de áudio (energia, dançabilidade, acusticidade, etc.).

A resposta é: **em parte**. E o motivo pelo qual não é 100% é, em si, o achado mais interessante do projeto.

O pipeline cobre o ciclo completo de um projeto de dados:

```
Ingestão → Limpeza → Análise Exploratória → Modelagem → Dashboard interativo
```

## 🗂️ Sobre os dados

- **114.000 faixas**, 125 gêneros originais do Spotify
- Após limpeza (remoção de duplicatas, outliers de duração, valores nulos): **~81.000 faixas**
- Gêneros originais agrupados em **12 macro-categorias** (ex: `death-metal`, `black-metal`, `metalcore` → `metal`) para viabilizar a modelagem

| Etapa | Linhas | Ação |
|---|---|---|
| Bruto | 114.000 | Dataset original do Kaggle |
| Duplicatas removidas | -32.656 | Mesma faixa em múltiplos álbuns/playlists |
| Nulos removidos | -1 | Campos essenciais ausentes |
| Outliers removidos | -86 | Faixas com duração < 30s ou > 20min |
| **Final** | **81.257** | Base limpa usada na análise |

## 💡 Principais achados

**1. Energia e acusticidade são quase espelhadas** (correlação de -0.73) — quanto mais "acústica" a faixa, menos energética ela soa. Faz sentido fisicamente, mas o tamanho da correlação surpreende.

**2. Metal é sonoramente isolado** — mediana de energia de 0.93, bem acima do segundo colocado (electronic, 0.77). É o gênero mais "extremo" do dataset em termos de intensidade sonora.

**3. Hip-hop domina popularidade apesar de ser o gênero com menos faixas** — mediana de popularidade de 57 (o segundo colocado, jazz/blues, tem 43), mesmo representando só 1,3% do dataset limpo. Isso sugere que as faixas de hip-hop presentes são majoritariamente hits mainstream, não uma amostra representativa do gênero como um todo.

**4. Faixas explícitas são discretamente mais populares** (popularidade média 38.7 vs 34.3) — contraintuitivo pra quem assume que conteúdo explícito limita alcance comercial.

**5. Pop e rock são os gêneros mais "ambíguos" sonoramente** — no modelo de classificação, são os que mais se confundem com outros gêneros, o que reflete seu papel como categorias "guarda-chuva" na indústria musical.

![Distribuição de gêneros](outputs/figures/genre_distribution.png)
![Correlação entre features](outputs/figures/feature_correlation.png)

## 🤖 Modelagem

**Decisão metodológica central:** o modelo foi treinado **sem** a variável `popularity`.

Incluir popularidade aumentaria a acurácia artificialmente — mas o modelo aprenderia padrões de mercado (quais gêneros vendem mais), não padrões sonoros. Como a pergunta do projeto é "dá pra reconhecer o gênero pelo som?", incluir popularidade responderia uma pergunta diferente e menos interessante.

| Modelo | Acurácia | Observação |
|---|---|---|
| Logistic Regression (baseline) | 33.5% | Baseline linear simples |
| **Random Forest** | **40.2%** | Modelo final, `class_weight='balanced'` |

O resultado de ~40% é **honesto, não uma limitação do projeto**: gênero musical é em parte uma categoria cultural e de marketing, não puramente acústica. Existe um teto teórico para esse tipo de classificação — e mapear onde esse teto está (quais gêneros são "reconhecíveis" e quais não são) é o próprio insight.

- **Metal e classical**: melhor precisão/recall — sonoramente distintos
- **Pop e rock**: pior desempenho — se confundem com quase tudo
- **Feature mais importante**: acusticidade, seguida de dançabilidade e energia

![Matriz de confusão](outputs/figures/confusion_matrix_random_forest.png)
![Importância das features](outputs/figures/feature_importance.png)

## 🖥️ Dashboard interativo

Construído em Streamlit, com 3 áreas:

- **Exploração detalhada**: filtros por gênero/popularidade/duração, correlação dinâmica, top faixas
- **Preditor de gênero**: sliders de características sonoras → previsão em tempo real + probabilidade por gênero
- **Desempenho do modelo**: matriz de confusão, importância de features, e um "sorteador" que compara previsões do modelo com o gênero real em faixas aleatórias

## 🛠️ Stack técnica

| Categoria | Ferramentas |
|---|---|
| Processamento | Python, Pandas, DuckDB |
| Machine Learning | scikit-learn (Random Forest, Logistic Regression) |
| Visualização | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Armazenamento | Parquet + DuckDB local |

## 📁 Estrutura do projeto

```
spotify-genre-analytics/
├── data/
│   ├── raw/                  # CSV original
│   ├── processed/            # Parquet limpo
│   └── spotify.duckdb        # Banco local
├── src/
│   ├── ingest.py             # CSV → DuckDB
│   ├── clean.py              # Limpeza e mapeamento de gêneros
│   ├── eda.py                # Análise exploratória
│   ├── features.py           # Preparação de features
│   ├── train_model.py        # Treino e avaliação dos modelos
│   └── utils.py
├── notebooks/
│   └── 01_eda.ipynb
├── app/
│   └── streamlit_app.py      # Dashboard interativo
├── models/
│   ├── genre_classifier.pkl
│   ├── scaler.pkl
│   └── labels.pkl
├── outputs/
│   └── figures/
├── requirements.txt
└── README.md
```

## 🚀 Como rodar

```bash
# 1. Clone o repositório
git clone https://github.com/TayschreN/spotify-genre-analytics.git
cd spotify-genre-analytics

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Baixe o dataset do Kaggle e coloque em data/raw/spotify_tracks.csv
# https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

# 4. Rode o pipeline
cd src
python ingest.py
python clean.py
python eda.py
python train_model.py

# 5. Rode o dashboard
cd ..
streamlit run app/streamlit_app.py
```

## 👤 Autor

**Gabriel Silva** — Data Engineering & Science, UFABC

- GitHub: [@TayschreN](https://github.com/TayschreN)
- LinkedIn: [gabrielfranca123](https://linkedin.com/in/gabrielfranca123)
- Portfólio: [datascienceportfol.io/gabrielfsilva2609](https://datascienceportfol.io/gabrielfsilva2609)

*Esse projeto faz parte de um portfólio em construção com foco em análise de dados e engenharia de dados. Feedback é bem-vindo!*