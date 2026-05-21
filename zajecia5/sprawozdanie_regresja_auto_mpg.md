# Sprawozdanie z laboratorium: porównanie algorytmów regresji

**Temat:** Przewidywanie zużycia paliwa samochodu (MPG) na podstawie cech technicznych
**Zbiór danych:** Auto MPG (UCI Machine Learning Repository), 398 pojazdów z lat 1970–1982
**Narzędzia:** Python 3.13, scikit-learn 1.7, pandas, numpy, matplotlib
**Data wykonania:** 2026-05-21

---

## Zadanie 1 — Uruchomienie programu

Program `regresja_auto_mpg_lab.py` został uruchomiony poleceniem `python regresja_auto_mpg_lab.py`.
Skrypt pobrał dane bezpośrednio z internetu, wytrenował 8 modeli i **poprawnie utworzył katalog
`wyniki_regresji`**. W katalogu pojawiły się pliki:

- `wyniki_modeli_regresji.csv` — tabela porównawcza metryk,
- `porownanie_modeli_rmse.png` — wykres słupkowy RMSE,
- `predykcje_vs_rzeczywiste.png` — wykres predykcji najlepszego modelu,
- `reszty_najlepszego_modelu.png` — wykres reszt,
- `waznosc_cech_random_forest.png` oraz `waznosc_cech_random_forest.csv` — ważność cech.

Dane wczytały się poprawnie: 398 wierszy, 9 kolumn. W kolumnie `horsepower` wykryto **6 braków danych**
(oznaczonych znakiem `?`), które pipeline uzupełnia medianą.

---

## Zadanie 2 — Model z najniższym RMSE_test

Tabela wyników (`wyniki_modeli_regresji.csv`), posortowana rosnąco według RMSE na zbiorze testowym:

| Model | MAE_test | RMSE_test | R2_test | RMSE_CV_mean | RMSE_CV_std |
|---|---|---|---|---|---|
| **SVR RBF** | **1.749** | **2.290** | **0.908** | 2.597 | 0.526 |
| Random Forest | 1.681 | 2.425 | 0.896 | 2.778 | 0.452 |
| k-NN | 2.019 | 2.572 | 0.883 | 2.964 | 0.499 |
| Gradient Boosting | 1.924 | 2.613 | 0.880 | 2.804 | 0.461 |
| Regresja liniowa | 2.110 | 2.704 | 0.871 | 3.198 | 0.376 |
| Ridge | 2.128 | 2.706 | 0.871 | 3.183 | 0.378 |
| Lasso | 2.127 | 2.714 | 0.870 | 3.190 | 0.378 |
| Drzewo decyzyjne | 2.176 | 3.290 | 0.809 | 3.844 | 0.573 |

**Model z najniższym RMSE_test to SVR RBF** (RMSE = 2.290). Co istotne, ten sam model ma także
najniższy średni błąd w walidacji krzyżowej (RMSE_CV = 2.597), co potwierdza, że jego przewaga
nie jest przypadkiem konkretnego podziału danych. Najsłabszy okazał się pojedynczy drzewo decyzyjne.

---

## Zadanie 3 — Interpretacja metryk MAE, RMSE i R²

- **MAE (Mean Absolute Error)** — średni błąd bezwzględny, czyli średnia z `|wartość rzeczywista − predykcja|`.
  Wyrażony w jednostkach zmiennej docelowej (tu: MPG). Dla SVR RBF MAE ≈ 1.75 oznacza, że model
  myli się średnio o ok. 1,75 mili na galon. MAE traktuje wszystkie błędy jednakowo.

- **RMSE (Root Mean Squared Error)** — pierwiastek ze średniego błędu kwadratowego. Również wyrażony
  w MPG, ale **mocniej karze duże błędy**, ponieważ błędy są podnoszone do kwadratu. RMSE jest zawsze
  ≥ MAE; im większa różnica między nimi, tym więcej w danych pojedynczych dużych pomyłek (odstających).

- **R² (współczynnik determinacji)** — bezwymiarowa miara mówiąca, **jaką część zmienności zmiennej
  docelowej model wyjaśnia**. R² = 1 to dopasowanie idealne, R² = 0 to model nie lepszy niż przewidywanie
  średniej. Dla SVR RBF R² = 0.908 oznacza, że model wyjaśnia ok. 91% zmienności zużycia paliwa.

Podsumowanie: MAE i RMSE mierzą wielkość błędu (mniej = lepiej), R² mierzy jakość dopasowania (więcej = lepiej).

---

## Zadanie 4 — Wykres `porownanie_modeli_rmse.png`

Wykres słupkowy przedstawia RMSE każdego modelu, posortowane rosnąco (najkrótszy słupek na górze).
**Ranking z wykresu jest w pełni zgodny z tabelą wyników:** od najlepszego SVR RBF, przez Random Forest,
k-NN, Gradient Boosting, regresję liniową, Ridge i Lasso, aż po najsłabsze drzewo decyzyjne.
Widać, że trzy modele liniowe (Regresja liniowa, Ridge, Lasso) dają praktycznie identyczne słupki —
to oczekiwane, bo różnią się tylko sposobem regularyzacji.

---

## Zadanie 5 — Wykres `predykcje_vs_rzeczywiste.png`

Wykres pokazuje punkty (rzeczywiste MPG, przewidywane MPG) dla najlepszego modelu (SVR RBF).
**Linia przerywana to linia idealnej predykcji `y = x`** — gdyby model nie popełniał żadnego błędu,
wszystkie punkty leżałyby dokładnie na niej. Punkty powyżej linii oznaczają zawyżenie predykcji,
punkty poniżej — zaniżenie. Na wykresie punkty układają się ciasno wzdłuż linii, co potwierdza dobre
dopasowanie (R² ≈ 0.91). Lekko większy rozrzut widać dla samochodów o wysokim MPG (powyżej 30).

---

## Zadanie 6 — Wykres `reszty_najlepszego_modelu.png`

Wykres przedstawia reszty (`rzeczywiste − przewidywane`) w funkcji wartości przewidywanej.
**Reszty są rozproszone w miarę równomiernie wokół linii zera**, bez wyraźnego trendu czy „lejka”
(kształtu rozszerzającego się). To dobry znak — oznacza, że model nie zawyża ani nie zaniża predykcji
systematycznie, a wariancja błędu jest zbliżona w całym zakresie. Widoczne są pojedyncze punkty
odstające (reszta ok. −7,5 oraz +6,8), ale stanowią one wyjątki, a nie regułę.

---

## Zadanie 7 — Zmiana parametrów modeli

Przygotowano skrypt `eksperymenty_modyfikacje.py`, który testuje różne ustawienia parametrów.
Wyniki (plik `wyniki_regresji/zadanie7_zmiana_parametrow.csv`):

| Wariant | MAE_test | RMSE_test | R2_test | RMSE_CV_mean |
|---|---|---|---|---|
| k-NN (n_neighbors=3) | 1.907 | 2.544 | 0.886 | 3.000 |
| k-NN (n_neighbors=5) — domyślny | 2.019 | 2.572 | 0.883 | 2.964 |
| k-NN (n_neighbors=15) | 2.056 | 2.728 | 0.869 | 3.179 |
| Drzewo (max_depth=3) | 2.684 | 3.642 | 0.766 | 3.692 |
| Drzewo (max_depth=5) — domyślny | 2.176 | 3.290 | 0.809 | 3.844 |
| Drzewo (max_depth=None) | 2.274 | 3.444 | 0.791 | 3.943 |
| Random Forest (n_estimators=10) | 1.807 | 2.426 | 0.896 | 2.820 |
| Random Forest (n_estimators=300) — domyślny | 1.681 | 2.425 | 0.896 | 2.778 |

**Wnioski:**

- **k-NN:** zmniejszenie liczby sąsiadów do 3 nieznacznie poprawiło wynik na zbiorze testowym
  (RMSE 2.572 → 2.544), ale pogorszyło wynik walidacji krzyżowej (CV 2.964 → 3.000). Zwiększenie do 15
  sąsiadów wyraźnie pogorszyło oba wyniki — model za bardzo „wygładza” predykcje (niedouczenie).
  Domyślne k = 5 jest dobrym kompromisem.
- **Drzewo decyzyjne:** zbyt płytkie drzewo (max_depth=3) jest niedouczone (RMSE 3.642), a drzewo
  bez limitu głębokości jest przeuczone (RMSE 3.444, gorzej niż depth=5). Domyślna głębokość 5 daje
  najlepszy wynik — to dobra ilustracja kompromisu bias–variance.
- **Random Forest:** liczba drzew 10 vs 300 daje niemal identyczny RMSE_test (2.426 vs 2.425),
  ale 300 drzew obniża MAE (1.807 → 1.681) i RMSE_CV (2.820 → 2.778). Więcej drzew daje stabilniejszy,
  mniej „głośny” wynik, choć przyrost jakości po pewnym progu jest niewielki.

---

## Zadanie 8 — Nowy model oraz modyfikacja cech

### 8a. Dodanie nowego modelu — Extra Trees Regressor

Dodano model **Extra Trees** (`ExtraTreesRegressor`, n_estimators=300) — wariant lasu losowego,
w którym progi podziału węzłów są dobierane losowo, co dodatkowo zmniejsza wariancję.

| Model | MAE_test | RMSE_test | R2_test | RMSE_CV_mean | RMSE_CV_std |
|---|---|---|---|---|---|
| Extra Trees (n_estimators=300) | 1.748 | 2.524 | 0.888 | 2.732 | 0.382 |

Wynik **plasuje się między Random Forest a k-NN**. Na zbiorze testowym Extra Trees jest nieco gorszy
od Random Forest (RMSE 2.524 vs 2.425), ale ma niższy i **bardziej stabilny RMSE w walidacji krzyżowej**
(CV 2.732 przy odchyleniu 0.382 — najniższym ze wszystkich modeli). To przyzwoity, stabilny model,
choć nie pobił lidera SVR RBF.

### 8b. Usunięcie cechy wejściowej — `displacement`

Sprawdzono, jak na model Random Forest wpływa usunięcie cechy `displacement` (pojemność silnika —
według wykresu ważności najważniejszej cechy):

| Wariant | MAE_test | RMSE_test | R2_test | RMSE_CV_mean |
|---|---|---|---|---|
| RF — wszystkie cechy | 1.681 | 2.425 | 0.896 | 2.778 |
| RF — bez `displacement` | 1.712 | 2.282 | 0.908 | 2.767 |

**Wynik nieznacznie się poprawił** (RMSE_test 2.425 → 2.282, R² 0.896 → 0.908), a wynik walidacji
krzyżowej pozostał praktycznie bez zmian (2.778 → 2.767). Może to wydawać się zaskakujące, skoro
`displacement` była cechą o najwyższej ważności. Wyjaśnienie: pojemność silnika jest **silnie
skorelowana** z masą pojazdu, mocą i liczbą cylindrów. Informacja niesiona przez `displacement`
jest więc w dużej mierze powielona przez pozostałe cechy, dlatego jej usunięcie nie szkodzi —
a usunięcie współliniowości wręcz lekko ustabilizowało model na tym konkretnym podziale danych.
Wniosek: wysoka ważność cechy nie oznacza, że jest ona niezastąpiona.

---

## Zadanie 9 — Trzy najważniejsze cechy (Random Forest)

Na podstawie pliku `waznosc_cech_random_forest.png` / `.csv`, trzy najważniejsze cechy dla
przewidywania MPG to:

1. **`displacement` (pojemność silnika)** — ważność 0.408,
2. **`weight` (masa pojazdu)** — ważność 0.178,
3. **`cylinders = 4` (czterocylindrowy silnik)** — ważność 0.145.

Tuż za podium znalazły się `model_year` (0.125) i `horsepower` (0.107). Wynik jest zgodny z intuicją:
duży, ciężki silnik o dużej pojemności spala więcej paliwa (niskie MPG), a małe, czterocylindrowe
silniki są ekonomiczne. Cecha `origin` (region pochodzenia) ma znikomą ważność.

---

## Zadanie 10 — Podsumowanie i rekomendacja modelu

**Rekomendowany model: SVR RBF (regresja wektorów nośnych z jądrem RBF).**

Uzasadnienie:

- Osiągnął **najniższy RMSE_test (2.290)** oraz **najwyższy R² (0.908)** spośród wszystkich 8 modeli.
- Co ważniejsze, ma także **najniższy średni RMSE w 5-krotnej walidacji krzyżowej (2.597)** —
  jego przewaga jest powtarzalna, a nie wynika ze szczęśliwego podziału danych.
- Zależność MPG od cech technicznych jest nieliniowa, dlatego model z jądrem RBF radzi sobie lepiej
  niż modele liniowe (Regresja liniowa / Ridge / Lasso, RMSE ≈ 2.70).

Zastrzeżenia praktyczne:

- SVR jest **wrażliwy na skalowanie** i parametry `C`, `gamma`, `epsilon` — działa dobrze tylko dzięki
  standaryzacji w pipeline; przy zmianie danych wymaga ponownego strojenia.
- Jeśli priorytetem jest **interpretowalność**, lepszym wyborem jest **Random Forest** (RMSE 2.425,
  R² 0.896) — tylko nieznacznie słabszy, a dodatkowo dostarcza ważność cech.
- Średni błąd predykcji rzędu 1,7–2,3 MPG przy wartościach docelowych 10–45 MPG to wynik bardzo dobry,
  w pełni wystarczający do zastosowań praktycznych.

**Rekomendacja końcowa:** do maksymalnej dokładności — SVR RBF; do wdrożenia wymagającego
interpretacji i odporności — Random Forest.

---

## Odpowiedzi na pytania kontrolne

**Dlaczego nie oceniamy modeli wyłącznie na zbiorze treningowym?**
Bo model zwykle „pamięta” dane treningowe i wynik na nich jest zbyt optymistyczny (ryzyko przeuczenia).
Realną zdolność uogólniania pokazuje dopiero ocena na danych, których model nie widział podczas treningu —
stąd zbiór testowy i walidacja krzyżowa.

**Jaka jest różnica między MAE a RMSE?**
MAE to średni błąd bezwzględny — traktuje wszystkie pomyłki jednakowo. RMSE podnosi błędy do kwadratu,
więc **mocniej karze duże błędy** i jest bardziej czuły na obserwacje odstające. RMSE ≥ MAE zawsze;
duża różnica między nimi sygnalizuje obecność pojedynczych dużych pomyłek.

**Co oznacza wysoka wartość R²?**
Że model wyjaśnia dużą część zmienności zmiennej docelowej. R² bliskie 1 oznacza bardzo dobre
dopasowanie; R² = 0 — model nie lepszy niż przewidywanie samej średniej; wartości ujemne — model
gorszy niż średnia.

**Dlaczego skalowanie cech jest ważne dla k-NN i SVR?**
Oba algorytmy opierają się na odległościach w przestrzeni cech. Bez skalowania cecha o dużych
wartościach (np. `weight` rzędu tysięcy) zdominowałaby cechę o małych wartościach (np. `acceleration`),
zniekształcając pojęcie „bliskości”. Standaryzacja sprowadza wszystkie cechy do porównywalnej skali.

**Dlaczego drzewo decyzyjne może się przeuczyć?**
Drzewo bez ograniczeń dzieli przestrzeń cech tak długo, aż dopasuje się do pojedynczych obserwacji
treningowych (łącznie z szumem). Powstają wtedy bardzo szczegółowe reguły, które nie uogólniają się
na nowe dane. Widać to w zadaniu 7: drzewo bez limitu głębokości miało gorszy RMSE niż drzewo
z `max_depth=5`.

**Czym Random Forest różni się od pojedynczego drzewa decyzyjnego?**
Random Forest trenuje wiele drzew na losowych podpróbkach danych i losowych podzbiorach cech,
a wynik uśrednia. Dzięki temu redukuje wariancję i przeuczenie pojedynczego drzewa — w tabeli
wyników RMSE spadł z 3.290 (drzewo) do 2.425 (las).

**Po czym poznać, że model systematycznie zawyża lub zaniża predykcje?**
Po wykresie reszt: jeśli reszty są przesunięte w jedną stronę względem zera lub układają się we
wzór (trend, krzywa) zamiast losowej chmury wokół zera, model ma błąd systematyczny (obciążenie).
W naszym przypadku reszty SVR RBF są rozproszone symetrycznie wokół zera — bez błędu systematycznego.

---

## Pliki wynikowe

| Plik | Opis |
|---|---|
| `wyniki_regresji/wyniki_modeli_regresji.csv` | Tabela metryk 8 modeli (zad. 1–4) |
| `wyniki_regresji/porownanie_modeli_rmse.png` | Wykres RMSE (zad. 4) |
| `wyniki_regresji/predykcje_vs_rzeczywiste.png` | Predykcje vs wartości rzeczywiste (zad. 5) |
| `wyniki_regresji/reszty_najlepszego_modelu.png` | Analiza reszt (zad. 6) |
| `wyniki_regresji/waznosc_cech_random_forest.png` / `.csv` | Ważność cech (zad. 9) |
| `wyniki_regresji/zadanie7_zmiana_parametrow.csv` | Wyniki zmiany parametrów (zad. 7) |
| `wyniki_regresji/zadanie8_modyfikacja_cech.csv` | Wyniki nowego modelu i usunięcia cechy (zad. 8) |
| `eksperymenty_modyfikacje.py` | Skrypt z eksperymentami do zadań 7 i 8 |
