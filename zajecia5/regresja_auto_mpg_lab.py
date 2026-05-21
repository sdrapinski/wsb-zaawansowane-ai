"""
Laboratorium: Porównanie algorytmów regresji w uczeniu nadzorowanym

Problem: przewidywanie zużycia paliwa samochodu, czyli wartości MPG
        (miles per gallon), na podstawie cech technicznych pojazdu.

Dane: Auto MPG z UCI Machine Learning Repository.
Źródło danych: https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data

Jak uruchomić:
    python regresja_auto_mpg_lab.py

Wymagane biblioteki:
    pip install pandas numpy matplotlib scikit-learn

Po uruchomieniu program zapisuje wyniki i wykresy w katalogu ./wyniki_regresji
"""

from __future__ import annotations

from pathlib import Path
from urllib.error import URLError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# -----------------------------------------------------------------------------
# 1. Ustawienia eksperymentu
# -----------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.25
OUTPUT_DIR = Path("wyniki_regresji")
OUTPUT_DIR.mkdir(exist_ok=True)

# Publiczny adres pliku z danymi. Program pobiera dane bezpośrednio z internetu.
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)

# Nazwy kolumn zgodne z opisem zbioru Auto MPG.
COLUMN_NAMES = [
    "mpg",           # zmienna docelowa: zużycie paliwa w milach na galon
    "cylinders",     # liczba cylindrów
    "displacement",  # pojemność silnika
    "horsepower",    # moc silnika; w danych występują braki oznaczone znakiem '?'
    "weight",        # masa pojazdu
    "acceleration",  # przyspieszenie
    "model_year",    # rok modelowy, np. 70 oznacza 1970
    "origin",        # kraj/region pochodzenia zakodowany liczbą: 1, 2, 3
    "car_name",      # nazwa samochodu; w tym ćwiczeniu nie używamy jej jako cechy
]


# -----------------------------------------------------------------------------
# 2. Funkcje pomocnicze
# -----------------------------------------------------------------------------


def create_one_hot_encoder() -> OneHotEncoder:
    """
    Tworzy OneHotEncoder w sposób zgodny z różnymi wersjami scikit-learn.

    W nowszych wersjach scikit-learn parametr nazywa się sparse_output,
    a w starszych sparse. Dzięki try/except kod jest bardziej przenośny.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """
    Pobiera dane Auto MPG z internetu i zwraca ramkę danych pandas.

    Plik źródłowy jest rozdzielony zmienną liczbą spacji. Kolumna car_name
    jest zapisana w cudzysłowie, dlatego używamy parametru quotechar='"'.
    Znak '?' interpretujemy jako brak danych.
    """
    data = pd.read_csv(
        url,
        sep=r"\s+",
        names=COLUMN_NAMES,
        na_values="?",
        quotechar='"',
    )
    return data



def build_preprocessor() -> ColumnTransformer:
    """
    Buduje część pipeline odpowiedzialną za przygotowanie danych.

    Pipeline dla cech numerycznych:
    1. Uzupełnia braki medianą.
    2. Standaryzuje dane, czyli przekształca je do skali o średniej 0
       i odchyleniu standardowym 1.

    Pipeline dla cech kategorycznych:
    1. Uzupełnia braki najczęstszą wartością.
    2. Koduje kategorie metodą one-hot encoding.
    """
    numeric_features = [
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
    ]

    categorical_features = [
        "cylinders",
        "origin",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", create_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor



def build_models() -> dict[str, Pipeline]:
    """
    Definiuje modele regresyjne porównywane w laboratorium.

    Każdy model jest opakowany w Pipeline:
        przygotowanie danych -> algorytm regresji

    Każdy model dostaje własną kopię preprocessora o tej samej konfiguracji.
    Dzięki temu mamy pewność, że identyczne przetwarzanie danych jest stosowane
    podczas treningu, testowania i walidacji krzyżowej.
    """
    regressors = {
        "Regresja liniowa": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.01, max_iter=20_000, random_state=RANDOM_STATE),
        "k-NN": KNeighborsRegressor(n_neighbors=5),
        "Drzewo decyzyjne": DecisionTreeRegressor(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "SVR RBF": SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1),
    }

    models = {}
    for model_name, regressor in regressors.items():
        models[model_name] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", regressor),
            ]
        )
    return models



def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float | str]:
    """
    Trenuje pojedynczy model i oblicza metryki jakości na zbiorze testowym.

    MAE  - średni błąd bezwzględny; niższy wynik jest lepszy.
    RMSE - pierwiastek ze średniego błędu kwadratowego; mocniej karze duże błędy.
    R2   - współczynnik determinacji; bliżej 1 oznacza lepsze dopasowanie.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model_name,
        "MAE_test": mae,
        "RMSE_test": rmse,
        "R2_test": r2,
    }



def cross_validate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[float, float]:
    """
    Wykonuje 5-krotną walidację krzyżową i zwraca średni RMSE oraz odchylenie.

    scikit-learn dla metryk błędu często używa wartości ujemnych, ponieważ
    wewnętrznie zakłada, że większy wynik jest lepszy. Dlatego wynik
    'neg_root_mean_squared_error' mnożymy przez -1.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rmse_scores = -scores["test_score"]
    return float(rmse_scores.mean()), float(rmse_scores.std())



def plot_model_comparison(results: pd.DataFrame) -> None:
    """Tworzy wykres porównujący modele według RMSE na zbiorze testowym."""
    sorted_results = results.sort_values("RMSE_test", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_results["model"], sorted_results["RMSE_test"])
    plt.xlabel("RMSE na zbiorze testowym")
    plt.ylabel("Model")
    plt.title("Porównanie modeli regresji - niższy RMSE jest lepszy")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "porownanie_modeli_rmse.png", dpi=150)
    plt.close()



def plot_predictions(best_model_name: str, best_model: Pipeline, X_test, y_test) -> None:
    """Tworzy wykres wartości rzeczywistych i przewidywanych dla najlepszego modelu."""
    y_pred = best_model.predict(X_test)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.75)
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("Rzeczywiste MPG")
    plt.ylabel("Przewidywane MPG")
    plt.title(f"Predykcje vs wartości rzeczywiste: {best_model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predykcje_vs_rzeczywiste.png", dpi=150)
    plt.close()



def plot_residuals(best_model_name: str, best_model: Pipeline, X_test, y_test) -> None:
    """Tworzy wykres reszt, czyli różnic: wartość rzeczywista - predykcja."""
    y_pred = best_model.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(9, 5))
    plt.scatter(y_pred, residuals, alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Przewidywane MPG")
    plt.ylabel("Reszta: rzeczywiste MPG - przewidywane MPG")
    plt.title(f"Analiza reszt: {best_model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reszty_najlepszego_modelu.png", dpi=150)
    plt.close()



def plot_feature_importance(random_forest_model: Pipeline) -> None:
    """
    Tworzy wykres ważności cech dla modelu Random Forest.

    Ważność cech jest dostępna tylko dla wybranych modeli, np. drzew i lasów.
    Dla regresji liniowej można analizować współczynniki, ale trzeba pamiętać
    o skali i kodowaniu zmiennych.
    """
    preprocessor = random_forest_model.named_steps["preprocessor"]
    regressor = random_forest_model.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    top_features = importance_df.head(12)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Ważność cechy")
    plt.ylabel("Cecha")
    plt.title("Najważniejsze cechy według modelu Random Forest")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "waznosc_cech_random_forest.png", dpi=150)
    plt.close()

    importance_df.to_csv(OUTPUT_DIR / "waznosc_cech_random_forest.csv", index=False)


# -----------------------------------------------------------------------------
# 3. Główny przebieg programu
# -----------------------------------------------------------------------------


def main() -> None:
    # Wczytanie danych z internetu.
    # Jeżeli uruchomienie zakończy się błędem sieci, sprawdź połączenie z internetem
    # albo pobierz plik danych ręcznie z adresu DATA_URL.
    try:
        data = load_data()
    except (URLError, OSError) as exc:
        raise SystemExit(
            "Nie udało się pobrać danych z internetu. "
            "Sprawdź połączenie sieciowe albo adres DATA_URL.\n"
            f"Szczegóły błędu: {exc}"
        ) from exc

    print("Pierwsze wiersze danych:")
    print(data.head())
    print("\nInformacje o danych:")
    print(data.info())
    print("\nLiczba braków danych w kolumnach:")
    print(data.isna().sum())

    # W tym ćwiczeniu przewidujemy mpg. Kolumnę car_name pomijamy,
    # ponieważ jest identyfikatorem/nazwą pojazdu, a nie typową cechą liczbową.
    target_column = "mpg"
    X = data.drop(columns=[target_column, "car_name"])
    y = data[target_column]

    # Podział na dane treningowe i testowe. Zbiór testowy służy tylko do oceny.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Definicja modeli do porównania. Każdy model zawiera identycznie skonfigurowany preprocessing.
    models = build_models()

    # Trenowanie i ocena każdego modelu.
    results = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"\nTrenuję model: {model_name}")

        test_results = evaluate_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        cv_rmse_mean, cv_rmse_std = cross_validate_model(model, X, y)
        test_results["RMSE_CV_mean"] = cv_rmse_mean
        test_results["RMSE_CV_std"] = cv_rmse_std

        results.append(test_results)
        fitted_models[model_name] = model

    results_df = pd.DataFrame(results).sort_values("RMSE_test", ascending=True)

    print("\nWyniki modeli posortowane według RMSE na zbiorze testowym:")
    print(results_df.round(3).to_string(index=False))

    # Zapis tabeli wyników do pliku CSV.
    results_df.to_csv(OUTPUT_DIR / "wyniki_modeli_regresji.csv", index=False)

    # Wybór najlepszego modelu według RMSE na zbiorze testowym.
    best_model_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    print(f"\nNajlepszy model według RMSE_test: {best_model_name}")

    # Wykresy diagnostyczne i porównawcze.
    plot_model_comparison(results_df)
    plot_predictions(best_model_name, best_model, X_test, y_test)
    plot_residuals(best_model_name, best_model, X_test, y_test)

    # Dodatkowa interpretacja dla Random Forest.
    # Ten wykres tworzymy niezależnie od tego, czy Random Forest jest najlepszy.
    plot_feature_importance(fitted_models["Random Forest"])

    print(f"\nPliki wynikowe zapisano w katalogu: {OUTPUT_DIR.resolve()}")
    print("Najważniejsze pliki:")
    print(" - wyniki_modeli_regresji.csv")
    print(" - porownanie_modeli_rmse.png")
    print(" - predykcje_vs_rzeczywiste.png")
    print(" - reszty_najlepszego_modelu.png")
    print(" - waznosc_cech_random_forest.png")


if __name__ == "__main__":
    main()
