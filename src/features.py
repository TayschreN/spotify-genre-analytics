"""
Prepara features e target pro modelo: split treino/teste, normalização,
encoding de categóricas. Não usa 'popularity' de propósito (ver README).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import get_connection

NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]
CATEGORICAL_FEATURES = ["key", "mode", "time_signature", "explicit"]
TARGET = "genre_macro"


def load_clean_data():
    con = get_connection()
    df = con.execute("SELECT * FROM tracks_clean WHERE genre_macro != 'other'").fetchdf()
    con.close()
    return df


def build_feature_matrix(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    X["explicit"] = X["explicit"].astype(int)
    y = df[TARGET]
    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
    X_test_scaled[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def prepare():
    df = load_clean_data()
    X, y = build_feature_matrix(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"Features usadas: {list(X.columns)}")
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    prepare()