"""
Treina classificador de macro-gênero usando só audio features (sem popularity).
Compara Logistic Regression (baseline) com Random Forest.
Salva o melhor modelo + scaler em models/.
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from features import prepare, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from utils import MODELS_DIR, ROOT

FIGURES_DIR = ROOT / "outputs" / "figures"


def train_baseline(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n=== {name} ===")
    print(f"Acurácia: {acc:.3f}")
    print(classification_report(y_test, preds))
    return preds, acc


def plot_confusion_matrix(y_test, preds, labels, name):
    cm = confusion_matrix(y_test, preds, labels=labels, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Matriz de confusão (normalizada) - {name}")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=120)
    plt.close()
    print(f"✅ Matriz de confusão salva ({name})")


def plot_feature_importance(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="teal")
    plt.title("Importância das features - Random Forest")
    plt.xlabel("Importância")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=120)
    plt.close()
    print("✅ feature_importance.png salvo")


def main():
    X_train, X_test, y_train, y_test, scaler = prepare()
    labels = sorted(y_train.unique())

    # Baseline
    baseline = train_baseline(X_train, y_train)
    preds_base, acc_base = evaluate(baseline, X_test, y_test, "Logistic Regression")
    plot_confusion_matrix(y_test, preds_base, labels, "Logistic Regression")

    # Random Forest
    rf = train_random_forest(X_train, y_train)
    preds_rf, acc_rf = evaluate(rf, X_test, y_test, "Random Forest")
    plot_confusion_matrix(y_test, preds_rf, labels, "Random Forest")
    plot_feature_importance(rf, NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    # Salva o melhor modelo
    best_model, best_name = (rf, "random_forest") if acc_rf >= acc_base else (baseline, "logistic_regression")
    joblib.dump(best_model, MODELS_DIR / "genre_classifier.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(labels, MODELS_DIR / "labels.pkl")

    print(f"\n✅ Melhor modelo: {best_name} (acc={max(acc_rf, acc_base):.3f})")
    print(f"✅ Salvo em {MODELS_DIR}")


if __name__ == "__main__":
    main()