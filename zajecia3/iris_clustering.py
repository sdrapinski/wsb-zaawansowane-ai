"""
============================================================
 UCZENIE NIENADZOROWANE NA ZBIORZE IRIS
 Algorytmy: K-Means (z metodą łokcia) oraz DBSCAN
============================================================
Autor: ćwiczenie laboratoryjne
Opis: Program demonstruje dwa klasyczne algorytmy klasteryzacji
      na dobrze known zbiorze Iris (150 próbek, 4 cechy, 3 gatunki).
      Ponieważ stosujemy UCZENIE NIENADZOROWANE, etykiety klas
      są używane WYŁĄCZNIE do oceny jakości klasteryzacji – 
      model ich nie widzi podczas trenowania.

Wymagane biblioteki:
    pip install scikit-learn matplotlib seaborn numpy pandas
"""

# ─────────────────────────────────────────────
# 1. IMPORTY
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,          # miara jakości klasteryzacji [-1, 1], im wyżej tym lepiej
    adjusted_rand_score,       # porównanie z prawdziwymi etykietami [0, 1]
    davies_bouldin_score,      # miara separacji klastrów, im niżej tym lepiej
    calinski_harabasz_score    # stosunek rozproszenia między/wewnątrz klastrów, im wyżej tym lepiej
)

# Styl wykresów
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

# Kolory dla maksymalnie 4 klastrów + szum (DBSCAN)
CLUSTER_COLORS = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12", "#7f8c8d"]
NOISE_COLOR    = "#bdc3c7"  # kolor dla punktów oznaczonych jako szum (-1) przez DBSCAN


# ─────────────────────────────────────────────
# 2. WCZYTANIE I PRZYGOTOWANIE DANYCH
# ─────────────────────────────────────────────
def load_and_prepare():
    """
    Wczytuje zbiór Iris i standaryzuje cechy.
    
    Standaryzacja (StandardScaler) przekształca każdą cechę tak,
    by miała średnią = 0 i odchylenie standardowe = 1.
    Jest KONIECZNA dla algorytmów opartych na odległości (K-Means, DBSCAN),
    bo bez niej cechy mierzone w różnych skalach dominują obliczenia dystansu.
    """
    iris = load_iris()
    X = iris.data          # (150, 4) – 4 cechy: długość/szerokość działki i płatka
    y = iris.target        # (150,)   – prawdziwe etykiety: 0, 1, 2
    feature_names = iris.feature_names
    target_names  = iris.target_names

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)   # fit() liczy średnią i std, transform() skaluje

    # PCA do 2 wymiarów – wyłącznie na potrzeby wizualizacji
    # PCA (Principal Component Analysis) szuka kierunków maksymalnej wariancji
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print("=" * 60)
    print("ZBIÓR IRIS – podstawowe informacje")
    print("=" * 60)
    print(f"  Liczba próbek:  {X.shape[0]}")
    print(f"  Liczba cech:    {X.shape[1]}")
    print(f"  Cechy:          {', '.join(feature_names)}")
    print(f"  Klasy:          {', '.join(target_names)}")
    print(f"  Wariancja wyjaśniona przez 2 pierwsze PC: "
          f"{pca.explained_variance_ratio_.sum():.1%}")
    print()

    return X_scaled, y, X_pca, feature_names, target_names, pca


# ─────────────────────────────────────────────
# 3. WIZUALIZACJA DANYCH WEJŚCIOWYCH
# ─────────────────────────────────────────────
def plot_raw_data(X_pca, y, target_names):
    """Wizualizuje dane w przestrzeni 2D (PCA) z prawdziwymi etykietami."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Zbiór Iris – wizualizacja wejściowa", fontweight="bold", fontsize=14)

    # Lewy wykres: PCA z prawdziwymi klasami (traktujemy jako "ground truth")
    for i, name in enumerate(target_names):
        mask = y == i
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        label=name, alpha=0.8, edgecolors="white", linewidths=0.4,
                        color=CLUSTER_COLORS[i], s=70)
    axes[0].set_title("Prawdziwe klasy (ground truth)")
    axes[0].set_xlabel("Pierwsza składowa główna (PC1)")
    axes[0].set_ylabel("Druga składowa główna (PC2)")
    axes[0].legend()

    # Prawy wykres: parplot cech oryginalnych (tylko 2 cechy dla czytelności)
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c="steelblue", alpha=0.5, edgecolors="white",
                    linewidths=0.3, s=60)
    axes[1].set_title("Dane bez etykiet (widok algorytmu)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].text(0.02, 0.97, "Algorytm NIE widzi kolorów –\nmusi sam odkryć strukturę",
                 transform=axes[1].transAxes, fontsize=9, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeeba", alpha=0.8))

    plt.tight_layout()
    plt.savefig("01_dane_wejsciowe.png", bbox_inches="tight")
    plt.show()
    print("[OK] Zapisano: 01_dane_wejsciowe.png\n")


# ─────────────────────────────────────────────
# 4. K-MEANS – METODA ŁOKCIA (ELBOW METHOD)
# ─────────────────────────────────────────────
def kmeans_elbow(X_scaled, k_range=range(1, 11)):
    """
    Metoda łokcia: uruchamia K-Means dla różnych wartości k,
    zbiera miarę inercji (WCSS) i rysuje wykres.

    PARAMETRY K-MEANS (sklearn.cluster.KMeans):
    -------------------------------------------
    n_clusters : int
        Liczba klastrów k – NAJWAŻNIEJSZY hiperparametr.
        Musi być podana z góry. Metoda łokcia pomaga go dobrać.

    init : {'k-means++', 'random'}
        Strategia inicjalizacji centroidów.
        'k-means++' (domyślna) – inteligentna inicjalizacja redukująca
        ryzyko złej zbieżności; losuje pierwsze centroidy z rozkładem
        proporcjonalnym do kwadratu odległości od już wybranych.
        'random' – losowe wybieranie punktów startowych.

    n_init : int (domyślnie 10 lub 'auto')
        Ile razy algorytm jest uruchamiany od nowa z różną inicjalizacją.
        Wybierany jest wynik z najmniejszą inercją. Więcej = stabilniej, wolniej.

    max_iter : int (domyślnie 300)
        Maksymalna liczba iteracji jednego uruchomienia algorytmu.
        Algorytm kończy wcześniej, gdy centroidy przestają się ruszać.

    tol : float (domyślnie 1e-4)
        Próg zbieżności – minimalna zmiana centroidów, poniżej której
        uznajemy, że algorytm się zbiegł.

    random_state : int
        Ziarno generatora losowego – zapewnia powtarzalność wyników.

    INERCJA (WCSS – Within-Cluster Sum of Squares):
        Suma kwadratów odległości każdego punktu od centroidu jego klastra.
        Im mniejsza, tym klastry są bardziej zwarte.
        Dla k=1 inercja jest maksymalna, dla k=n spada do 0.
        "Łokieć" wykresu wskazuje k, po którym spadek staje się mały.
    """
    inertias    = []  # WCSS dla każdego k
    silhouettes = []  # Silhouette Score dla każdego k (zaczyna się od k=2)

    for k in k_range:
        km = KMeans(
            n_clusters=k,
            init="k-means++",   # lepsza inicjalizacja niż losowa
            n_init=10,           # 10 niezależnych uruchomień
            max_iter=300,
            random_state=42
        )
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)

        # Silhouette Score wymaga minimum 2 klastrów i 2 unikalnych etykiet
        if k >= 2 and len(np.unique(labels)) > 1:
            silhouettes.append(silhouette_score(X_scaled, labels))
        else:
            silhouettes.append(np.nan)

    # ── Wykres metody łokcia ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("K-Means: metoda łokcia i Silhouette Score", fontweight="bold", fontsize=14)

    # Inercja
    axes[0].plot(list(k_range), inertias, marker="o", color="#e74c3c",
                 linewidth=2, markersize=7)
    axes[0].axvline(x=3, color="#2ecc71", linestyle="--", linewidth=1.5,
                    label="Optymalny łokieć (k=3)")
    # Zaznaczenie zmiany nachylenia – "łokcia"
    axes[0].annotate('„Łokieć"\n(k=3)', xy=(3, inertias[2]),
                     xytext=(5, inertias[2] + (inertias[0] - inertias[-1]) * 0.15),
                     arrowprops=dict(arrowstyle="->", color="gray"),
                     fontsize=10, color="gray")
    axes[0].set_xlabel("Liczba klastrów k")
    axes[0].set_ylabel("Inercja (WCSS)")
    axes[0].set_title("Metoda łokcia (Elbow Method)")
    axes[0].legend()

    # Silhouette Score
    k_sil = list(k_range)[1:]   # od k=2
    axes[1].plot(k_sil, silhouettes[1:], marker="s", color="#2980b9",
                 linewidth=2, markersize=7)
    best_k_sil = k_sil[int(np.nanargmax(silhouettes[1:]))]
    axes[1].axvline(x=best_k_sil, color="#e67e22", linestyle="--", linewidth=1.5,
                    label=f"Najwyższy Silhouette (k={best_k_sil})")
    axes[1].set_xlabel("Liczba klastrów k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Współczynnik Silhouette")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("02_elbow_silhouette.png", bbox_inches="tight")
    plt.show()
    print("[OK] Zapisano: 02_elbow_silhouette.png\n")

    print(f"  Inercje dla k=1..10: {[round(v, 1) for v in inertias]}")
    print(f"  Silhouette dla k=2..10: {[round(v, 3) for v in silhouettes[1:]]}")
    print()

    return inertias, silhouettes


# ─────────────────────────────────────────────
# 5. K-MEANS – FINALNY MODEL (k=3)
# ─────────────────────────────────────────────
def run_kmeans(X_scaled, X_pca, y, target_names, k=3):
    """
    Trenuje finalny model K-Means z k=3 i wizualizuje wyniki.
    """
    km = KMeans(
        n_clusters=k,          # 3 klastry – zgodnie z metodą łokcia
        init="k-means++",      # inteligentna inicjalizacja
        n_init=20,             # 20 uruchomień dla stabilności
        max_iter=300,
        tol=1e-4,
        random_state=42,
        verbose=0              # 0 = brak logów, 1 = logi iteracji
    )
    labels_km = km.fit_predict(X_scaled)

    # Centroidy w oryginalnej przestrzeni cech (po skalowaniu)
    centroids_scaled = km.cluster_centers_   # shape (k, n_features)
    # Rzutujemy centroidy na płaszczyznę PCA do wizualizacji
    centroids_pca = PCA(n_components=2, random_state=42).fit(X_scaled).transform(centroids_scaled)

    # ── Metryki jakości ──────────────────────────────────────────────
    sil  = silhouette_score(X_scaled, labels_km)
    dbi  = davies_bouldin_score(X_scaled, labels_km)
    chi  = calinski_harabasz_score(X_scaled, labels_km)
    ari  = adjusted_rand_score(y, labels_km)

    print("=" * 60)
    print(f"K-MEANS (k={k}) – metryki jakości")
    print("=" * 60)
    print(f"  Inercja (WCSS):          {km.inertia_:.2f}")
    print(f"  Silhouette Score:        {sil:.4f}   (im bliżej 1, tym lepiej)")
    print(f"  Davies-Bouldin Index:    {dbi:.4f}   (im niżej, tym lepiej)")
    print(f"  Calinski-Harabasz:       {chi:.2f}  (im wyżej, tym lepiej)")
    print(f"  Adjusted Rand Index:     {ari:.4f}   (porównanie z ground truth, 1=ideał)")
    print()

    # ── Wizualizacja ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle(f"K-Means (k={k}) – wyniki klasteryzacji", fontweight="bold", fontsize=15)

    # (A) Klastry w przestrzeni PCA
    ax_a = fig.add_subplot(gs[0, 0])
    for c in range(k):
        mask = labels_km == c
        ax_a.scatter(X_pca[mask, 0], X_pca[mask, 1],
                     color=CLUSTER_COLORS[c], label=f"Klaster {c}",
                     alpha=0.8, edgecolors="white", linewidths=0.4, s=70)
    # Centroidy w przestrzeni PCA – duże gwiazdki
    pca2 = PCA(n_components=2, random_state=42).fit(X_scaled)
    c_pca = pca2.transform(centroids_scaled)
    ax_a.scatter(c_pca[:, 0], c_pca[:, 1], marker="*", s=300,
                 color="black", zorder=5, label="Centroidy")
    ax_a.set_title("(A) Klastry K-Means (przestrzeń PCA)")
    ax_a.set_xlabel("PC1"); ax_a.set_ylabel("PC2")
    ax_a.legend()

    # (B) Porównanie z ground truth
    ax_b = fig.add_subplot(gs[0, 1])
    for i, name in enumerate(target_names):
        mask = y == i
        ax_b.scatter(X_pca[mask, 0], X_pca[mask, 1],
                     color=CLUSTER_COLORS[i], label=name,
                     alpha=0.8, edgecolors="white", linewidths=0.4, s=70)
    ax_b.set_title("(B) Prawdziwe klasy (ground truth)")
    ax_b.set_xlabel("PC1"); ax_b.set_ylabel("PC2")
    ax_b.legend()

    # (C) Macierz pomyłek (przypisanie klastrów do klas)
    ax_c = fig.add_subplot(gs[1, 0])
    conf = np.zeros((k, 3), dtype=int)
    for true, pred in zip(y, labels_km):
        conf[pred, true] += 1
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", ax=ax_c,
                xticklabels=target_names,
                yticklabels=[f"Klaster {i}" for i in range(k)])
    ax_c.set_title("(C) Macierz przypisania: klaster → klasa")
    ax_c.set_xlabel("Prawdziwa klasa")
    ax_c.set_ylabel("Przypisany klaster")

    # (D) Tabela metryk
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    metrics_data = [
        ["Metryka", "Wartość", "Interpretacja"],
        ["Inercja (WCSS)", f"{km.inertia_:.1f}", "Zwartość klastrów"],
        ["Silhouette", f"{sil:.4f}", "1 = ideał, <0 = źle"],
        ["Davies-Bouldin", f"{dbi:.4f}", "Bliżej 0 = lepiej"],
        ["Calinski-Harabasz", f"{chi:.1f}", "Wyżej = lepiej"],
        ["Adj. Rand Index", f"{ari:.4f}", "1 = idealne dopasowanie"],
    ]
    table = ax_d.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    ax_d.set_title("(D) Podsumowanie metryk", pad=20)

    plt.tight_layout()
    plt.savefig("03_kmeans_wyniki.png", bbox_inches="tight")
    plt.show()
    print("[OK] Zapisano: 03_kmeans_wyniki.png\n")

    return labels_km, km


# ─────────────────────────────────────────────
# 6. DBSCAN – ANALIZA PARAMETRÓW (k-distance)
# ─────────────────────────────────────────────
def dbscan_kdistance(X_scaled, k=5):
    """
    Wykres k-distance (k-odległości) – metoda doboru parametru eps dla DBSCAN.

    Dla każdego punktu liczymy odległość do jego k-tego najbliższego sąsiada.
    Sortujemy wyniki malejąco. "Łokieć" wykresu wskazuje dobrą wartość eps:
    - punkty przed łokciem są gęste (core points)
    - punkty za łokciem są rzadkie (szum lub border points)
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    # Kolumna k-1 to odległość do k-tego sąsiada (0-indexed, 0=punkt sam w sobie)
    k_distances = np.sort(distances[:, k - 1])[::-1]   # sortowanie malejące

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_distances, color="#8e44ad", linewidth=2)
    ax.axhline(y=0.5, color="#e74c3c", linestyle="--", linewidth=1.5,
               label="Sugerowane eps ≈ 0.5")
    ax.set_xlabel("Punkty posortowane malejąco wg odległości")
    ax.set_ylabel(f"Odległość do {k}. najbliższego sąsiada")
    ax.set_title(f"Wykres {k}-odległości – dobór parametru eps dla DBSCAN")
    ax.legend()
    ax.text(len(k_distances) * 0.6, 0.55, '← „Łokieć" wykresu\n   sugeruje wartość eps',
            fontsize=10, color="#e74c3c")

    plt.tight_layout()
    plt.savefig("04_dbscan_kdistance.png", bbox_inches="tight")
    plt.show()
    print("[OK] Zapisano: 04_dbscan_kdistance.png\n")


# ─────────────────────────────────────────────
# 7. DBSCAN – FINALNY MODEL
# ─────────────────────────────────────────────
def run_dbscan(X_scaled, X_pca, y, target_names):
    """
    Trenuje model DBSCAN i wizualizuje wyniki wraz z analizą wrażliwości na eps.

    PARAMETRY DBSCAN (sklearn.cluster.DBSCAN):
    ------------------------------------------
    eps : float (domyślnie 0.5)
        Promień sąsiedztwa – maksymalna odległość między dwoma punktami,
        przy której drugi jest uznawany za sąsiada pierwszego.
        To kluczowy hiperparametr. Zbyt małe eps → wszystko = szum.
        Zbyt duże eps → wszystko w jednym klastrze.
        Dobór: wykres k-odległości (patrz funkcja dbscan_kdistance).

    min_samples : int (domyślnie 5)
        Minimalna liczba punktów w sąsiedztwie eps, by punkt był uznany za
        "core point" (rdzeń klastra). Wpływa na odporność na szum.
        Zasada kciuka: min_samples ≈ wymiary_danych + 1
                       lub min_samples = 2 * n_features.
        Większe min_samples → bardziej restrykcyjne klastry, więcej szumu.

    metric : str lub callable (domyślnie 'euclidean')
        Miara odległości: 'euclidean', 'manhattan', 'cosine', itd.
        Dla znormalizowanych danych 'euclidean' jest standardowym wyborem.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}
        Algorytm wyszukiwania sąsiadów.
        'auto' wybiera najefektywniejszy dla danych.
        'brute' = siłowe O(n²), dobre dla małych zbiorów.
        'ball_tree'/'kd_tree' = efektywne dla dużych zbiorów, niskich wymiarów.

    n_jobs : int (domyślnie None)
        Liczba wątków CPU. -1 = użyj wszystkich dostępnych.

    TYPY PUNKTÓW W DBSCAN:
        ● Core point (rdzeń): ma co najmniej min_samples sąsiadów w eps.
        ● Border point (brzeg): jest sąsiadem core pointu, ale sam nie jest core.
        ● Noise point (szum): nie jest core ani border – etykieta = -1.
    """
    # Finalny model z dobranymi parametrami
    dbscan = DBSCAN(
        eps=0.5,              # promień sąsiedztwa – dobrany z wykresu k-odległości
        min_samples=5,        # min. punktów by być "core point"
        metric="euclidean",   # miara euklidesowa (dane znormalizowane)
        algorithm="auto",     # automatyczny wybór struktury danych
        n_jobs=-1             # wielowątkowość
    )
    labels_db = dbscan.fit_predict(X_scaled)

    n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_noise_db    = np.sum(labels_db == -1)

    print("=" * 60)
    print("DBSCAN – wyniki")
    print("=" * 60)
    print(f"  Liczba odkrytych klastrów: {n_clusters_db}")
    print(f"  Punkty szumu (etykieta=-1): {n_noise_db} ({100*n_noise_db/len(labels_db):.1f}%)")

    if n_clusters_db >= 2 and len(np.unique(labels_db[labels_db != -1])) >= 2:
        # Metryki tylko na punktach niebędących szumem
        mask_no_noise = labels_db != -1
        sil_db = silhouette_score(X_scaled[mask_no_noise], labels_db[mask_no_noise])
        ari_db = adjusted_rand_score(y, labels_db)
        print(f"  Silhouette Score (bez szumu): {sil_db:.4f}")
        print(f"  Adjusted Rand Index:          {ari_db:.4f}")
    else:
        print("  (za mało klastrów do obliczenia metryk)")

    print()

    # ── Wizualizacja główna ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("DBSCAN – wyniki klasteryzacji", fontweight="bold", fontsize=15)

    # Kolory: -1 (szum) = szary, klastry = kolory
    unique_labels = sorted(set(labels_db))
    color_map = {}
    ci = 0
    for lbl in unique_labels:
        if lbl == -1:
            color_map[lbl] = NOISE_COLOR
        else:
            color_map[lbl] = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
            ci += 1

    point_colors = [color_map[l] for l in labels_db]

    # Panel (A): DBSCAN w przestrzeni PCA
    for lbl in unique_labels:
        mask = labels_db == lbl
        label_str = "Szum (−1)" if lbl == -1 else f"Klaster {lbl}"
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        color=color_map[lbl], label=label_str,
                        alpha=0.8 if lbl != -1 else 0.4,
                        edgecolors="white", linewidths=0.3,
                        s=70 if lbl != -1 else 30,
                        marker="o" if lbl != -1 else "x")
    axes[0].set_title("(A) Klastry DBSCAN (PCA)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=8)

    # Panel (B): Ground truth dla porównania
    for i, name in enumerate(target_names):
        mask = y == i
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        color=CLUSTER_COLORS[i], label=name,
                        alpha=0.8, edgecolors="white", linewidths=0.3, s=70)
    axes[1].set_title("(B) Prawdziwe klasy (ground truth)")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].legend(fontsize=8)

    # Panel (C): Analiza wrażliwości na eps
    eps_values    = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]
    n_clusters_v  = []
    n_noise_v     = []
    for e in eps_values:
        lbl = DBSCAN(eps=e, min_samples=5).fit_predict(X_scaled)
        n_clusters_v.append(len(set(lbl)) - (1 if -1 in lbl else 0))
        n_noise_v.append(np.sum(lbl == -1))

    ax2 = axes[2].twinx()
    axes[2].bar([str(e) for e in eps_values], n_clusters_v,
                color="#2980b9", alpha=0.7, label="Liczba klastrów")
    ax2.plot([str(e) for e in eps_values], n_noise_v,
             color="#e74c3c", marker="o", linewidth=2, label="Punkty szumu")
    axes[2].set_xlabel("Wartość eps")
    axes[2].set_ylabel("Liczba klastrów", color="#2980b9")
    ax2.set_ylabel("Punkty szumu", color="#e74c3c")
    axes[2].set_title("(C) Wrażliwość DBSCAN na eps\n(min_samples=5)")
    axes[2].axvline(x="0.5", color="#27ae60", linestyle="--", linewidth=2,
                    label="Wybrane eps=0.5")
    lines1, lab1 = axes[2].get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, lab1 + lab2, fontsize=8)

    plt.tight_layout()
    plt.savefig("05_dbscan_wyniki.png", bbox_inches="tight")
    plt.show()
    print("[OK] Zapisano: 05_dbscan_wyniki.png\n")

    return labels_db, dbscan


# ─────────────────────────────────────────────
# 8. PORÓWNANIE K-MEANS vs DBSCAN
# ─────────────────────────────────────────────
def compare_algorithms(X_scaled, X_pca, y, labels_km, labels_db, target_names):
    """
    Zestawia wyniki obu algorytmów na jednym wykresie zbiorczym.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Porównanie: K-Means vs DBSCAN vs Ground Truth", fontweight="bold", fontsize=15)

    datasets = [
        (labels_km, "K-Means (k=3)", axes[0]),
        (labels_db, "DBSCAN (eps=0.5, min_samples=5)", axes[1]),
        (y,         "Ground Truth", axes[2]),
    ]

    for labels, title, ax in datasets:
        unique = sorted(set(labels))
        ci = 0
        for lbl in unique:
            mask = labels == lbl
            if lbl == -1:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           color=NOISE_COLOR, label="Szum", alpha=0.4, s=25, marker="x")
            else:
                name = target_names[lbl] if (title == "Ground Truth") else f"Klaster {lbl}"
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           color=CLUSTER_COLORS[ci], label=name,
                           alpha=0.8, edgecolors="white", linewidths=0.3, s=70)
                ci += 1
        ax.set_title(title)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=8)

    # Metryki zbiorcze
    ari_km = adjusted_rand_score(y, labels_km)
    ari_db = adjusted_rand_score(y, labels_db)

    mask_nn = labels_db != -1
    sil_km = silhouette_score(X_scaled, labels_km)
    sil_db = silhouette_score(X_scaled[mask_nn], labels_db[mask_nn]) if mask_nn.sum() > 0 else np.nan

    print("=" * 60)
    print("PORÓWNANIE ALGORYTMÓW")
    print("=" * 60)
    print(f"{'Metryka':<30} {'K-Means':>12} {'DBSCAN':>12}")
    print("-" * 55)
    print(f"{'Silhouette Score':<30} {sil_km:>12.4f} {sil_db:>12.4f}")
    print(f"{'Adjusted Rand Index':<30} {ari_km:>12.4f} {ari_db:>12.4f}")
    print(f"{'Liczba klastrów':<30} {'3':>12} {str(len(set(labels_db))-(1 if -1 in labels_db else 0)):>12}")
    print(f"{'Punkty szumu':<30} {'0':>12} {str(np.sum(labels_db==-1)):>12}")
    print()

    plt.tight_layout()
    plt.savefig("06_porownanie.png", bbox_inches="tight")
    plt.show()
    print("[OK] Zapisano: 06_porownanie.png\n")


# ─────────────────────────────────────────────
# 9. GŁÓWNA FUNKCJA PROGRAMU
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  UCZENIE NIENADZOROWANE – ZBIÓR IRIS")
    print("  K-Means + DBSCAN")
    print("=" * 60 + "\n")

    # --- Krok 1: Dane ---
    X_scaled, y, X_pca, feature_names, target_names, pca = load_and_prepare()

    # --- Krok 2: Wizualizacja danych wejściowych ---
    plot_raw_data(X_pca, y, target_names)

    # --- Krok 3: Metoda łokcia (dobór k dla K-Means) ---
    print(">>> K-Means: Metoda łokcia <<<")
    kmeans_elbow(X_scaled, k_range=range(1, 11))

    # --- Krok 4: Finalny K-Means ---
    print(">>> K-Means: Model finalny (k=3) <<<")
    labels_km, km_model = run_kmeans(X_scaled, X_pca, y, target_names, k=3)

    # --- Krok 5: DBSCAN – dobór eps ---
    print(">>> DBSCAN: Wykres k-odległości <<<")
    dbscan_kdistance(X_scaled, k=5)

    # --- Krok 6: Finalny DBSCAN ---
    print(">>> DBSCAN: Model finalny <<<")
    labels_db, dbscan_model = run_dbscan(X_scaled, X_pca, y, target_names)

    # --- Krok 7: Porównanie ---
    print(">>> Porównanie algorytmów <<<")
    compare_algorithms(X_scaled, X_pca, y, labels_km, labels_db, target_names)

    print("=" * 60)
    print("  Wszystkie wykresy zapisane. Program zakończony.")
    print("=" * 60)


if __name__ == "__main__":
    main()
