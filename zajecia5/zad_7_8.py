"""
Eksperymenty do zadań 7 i 8 z instrukcji laboratorium.

Zadanie 7: zmiana parametrów wybranych modeli i porównanie wyników.
Zadanie 8: dodanie nowego modelu regresji oraz test usunięcia cechy wejściowej.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from regresja_auto_mpg_lab import (
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    build_preprocessor,
    cross_validate_model,
    load_data,
)


def ocen(nazwa, regressor, X_train, X_test, y_train, y_test, X, y):
    """Buduje pipeline (preprocessing + model), trenuje i liczy metryki."""
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv_mean, cv_std = cross_validate_model(model, X, y)

    return {
        "wariant": nazwa,
        "MAE_test": mae,
        "RMSE_test": rmse,
        "R2_test": r2,
        "RMSE_CV_mean": cv_mean,
        "RMSE_CV_std": cv_std,
    }


def main() -> None:
    data = load_data()
    target_column = "mpg"
    X = data.drop(columns=[target_column, "car_name"])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    wyniki = []

    # =====================================================================
    # ZADANIE 7 - zmiana parametrów modeli
    # =====================================================================
    print("=" * 70)
    print("ZADANIE 7 - wplyw zmiany parametrow modeli")
    print("=" * 70)

    # k-NN: domyslnie n_neighbors=5; sprawdzamy 3 i 15
    for k in (3, 5, 15):
        wyniki.append(
            ocen(f"k-NN (n_neighbors={k})", KNeighborsRegressor(n_neighbors=k),
                 X_train, X_test, y_train, y_test, X, y)
        )

    # Drzewo decyzyjne: domyslnie max_depth=5; sprawdzamy 3 i None (bez limitu)
    for d in (3, 5, None):
        wyniki.append(
            ocen(f"Drzewo (max_depth={d})",
                 DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE),
                 X_train, X_test, y_train, y_test, X, y)
        )

    # Random Forest: domyslnie n_estimators=300; sprawdzamy 10 i 300
    for n in (10, 300):
        wyniki.append(
            ocen(f"Random Forest (n_estimators={n})",
                 RandomForestRegressor(n_estimators=n, random_state=RANDOM_STATE,
                                       n_jobs=-1),
                 X_train, X_test, y_train, y_test, X, y)
        )

    df7 = pd.DataFrame(wyniki)
    print(df7.round(3).to_string(index=False))
    df7.to_csv(OUTPUT_DIR / "zadanie7_zmiana_parametrow.csv", index=False)

    # =====================================================================
    # ZADANIE 8a - dodanie nowego modelu regresji (Extra Trees)
    # =====================================================================
    print("\n" + "=" * 70)
    print("ZADANIE 8a - nowy model: Extra Trees Regressor")
    print("=" * 70)

    nowy = ocen(
        "Extra Trees (n_estimators=300)",
        ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        X_train, X_test, y_train, y_test, X, y,
    )
    print(pd.DataFrame([nowy]).round(3).to_string(index=False))

    # =====================================================================
    # ZADANIE 8b - usuniecie jednej cechy wejsciowej (displacement)
    # =====================================================================
    print("\n" + "=" * 70)
    print("ZADANIE 8b - usuniecie cechy 'displacement' (Random Forest)")
    print("=" * 70)

    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)

    pelne = ocen("RF - wszystkie cechy", rf, X_train, X_test, y_train, y_test, X, y)

    # Wariant bez kolumny displacement
    X_bez = X.drop(columns=["displacement"])
    Xtr_bez, Xte_bez, ytr, yte = train_test_split(
        X_bez, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Preprocessor odwoluje sie do nazw kolumn, wiec budujemy go recznie
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from regresja_auto_mpg_lab import create_one_hot_encoder

    num = ["horsepower", "weight", "acceleration", "model_year"]
    cat = ["cylinders", "origin"]
    pre_bez = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), num),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", create_one_hot_encoder())]), cat),
        ]
    )
    model_bez = Pipeline([("preprocessor", pre_bez),
                          ("regressor", RandomForestRegressor(
                              n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))])
    model_bez.fit(Xtr_bez, ytr)
    yp = model_bez.predict(Xte_bez)
    cv_m, cv_s = cross_validate_model(model_bez, X_bez, y)
    bez = {
        "wariant": "RF - bez 'displacement'",
        "MAE_test": mean_absolute_error(yte, yp),
        "RMSE_test": np.sqrt(mean_squared_error(yte, yp)),
        "R2_test": r2_score(yte, yp),
        "RMSE_CV_mean": cv_m,
        "RMSE_CV_std": cv_s,
    }

    df8 = pd.DataFrame([pelne, bez])
    print(df8.round(3).to_string(index=False))
    df8.to_csv(OUTPUT_DIR / "zadanie8_modyfikacja_cech.csv", index=False)

    print(f"\nPliki wynikowe zapisano w katalogu: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
