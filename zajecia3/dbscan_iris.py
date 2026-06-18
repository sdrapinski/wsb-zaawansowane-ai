import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Wczytanie danych
iris = load_iris()
X = iris.data[:, :2]   # te same dwie cechy

print('Dane załadowane:')
print(f'Liczba próbek: {X.shape[0]}')
print(f'Cechy: {iris.feature_names[:2]}')

# Standaryzacja danych (ważna dla DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Parametry DBSCAN
eps = 0.5
min_samples = 5

# Trenowanie modelu
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
etykiety = dbscan.fit_predict(X_scaled)

# Liczba klastrów (bez szumu)
n_clusters = len(set(etykiety)) - (1 if -1 in etykiety else 0)

print(f'\nLiczba znalezionych klastrów: {n_clusters}')

# Liczba punktów w klastrach
print('\nLiczba próbek w klastrach:')
for i in set(etykiety):
    liczba = np.sum(etykiety == i)
    
    if i == -1:
        print(f'Szum (noise): {liczba} próbek')
    else:
        print(f'Klaster {i+1}: {liczba} próbek')

# Kolory
kolory = ['red','blue','green','orange','purple','black','brown']

# Rysowanie klastrów
for i in set(etykiety):

    if i == -1:
        kolor = 'grey'
        label = 'Szum'
    else:
        kolor = kolory[i]
        label = f'Klaster {i+1}'

    punkty = X[etykiety == i]

    plt.scatter(
        punkty[:,0],
        punkty[:,1],
        c=kolor,
        label=label,
        alpha=0.7,
        s=50
    )

plt.title('Klastrowanie DBSCAN')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()