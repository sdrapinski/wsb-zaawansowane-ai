import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data[:, :2]   # 2 pierwsze cechy

print('Dane załadowane:')
print(f'Liczba próbek: {X.shape[0]}')
print(f'Cechy {iris.feature_names[:2]}')

# 1. Metoda łokcia - znajdowanie optymalnej liczby klastrów
inercje = []
liczby_klastrow = range(1, 9)
for k in liczby_klastrow:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inercje.append(kmeans.inertia_)
    print(f'K={k}: inercja = {kmeans.inertia_:.2f}')

# Wykres metody łokcia
plt.plot(liczby_klastrow, inercje, 'bo-', linewidth=2, markersize=8)
plt.title('Metoda łokcia')
plt.xlabel('Liczba klastrów (K)')
plt.ylabel('Inercje')
plt.grid(True, alpha=0.3)
plt.show()

k_optymalne = 3
kmeans = KMeans(n_clusters=k_optymalne)
etykiety = kmeans.fit_predict(X)
centroidy = kmeans.cluster_centers_

print(f'Klastrowanie z K={k_optymalne}:')
print(f'Inercja: {kmeans.inertia_:.2f}')
print(f'Liczba iteracji: {kmeans.n_iter_}')

# Liczba próbek w każdym klastrze
print('\nLiczba próbek w klastrze:')
for i in range(k_optymalne):
    liczba = np.sum(etykiety == i)
    print(f'Klaster {i + 1}: {liczba} próbek')

# Kolory dla różnych klastrów:
kolory = ['red', 'blue', 'green', 'white', 'black', 'pink', 'yellow', 'orange', 'purple', 'grey']

# rysowanie punktów dla każdego klastra
for i in range(k_optymalne):
    punkty_klastra = X[etykiety == i]
    plt.scatter(punkty_klastra[:, 0], punkty_klastra[:, 1],
                c=kolory[i], label=f'Klaster {i+1}', alpha=0.7, s=50)

# rysowanie centroidów
plt.scatter(centroidy[:, 0], centroidy[:, 1],
            c='black', marker='x', s=300, linewidths=4, label='Centroidy')

# Numery centroidów
for i, centroid in enumerate(centroidy):
    plt.annotate(f'C{i+1}', (centroid[0], centroid[1]),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=12, fontweight='bold', color='white',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=kolory[i]))

plt.title(f'Wyniki klastrowania K-means(K={k_optymalne})')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()