import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv('iris.csv')
print(df)
print(df.describe().T.round(2).to_string())
print(df['class'].value_counts())
species = {
    'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica':2
}
df['class_value'] = df['class'].map(species)
print(df['class_value'].value_counts())
print(df)
sample = np.array([5.6, 3.2, 5.2, 1.45])

sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class')
plt.scatter(5.6, 3.2, c='r')
plt.show()
sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class')
plt.scatter(5.2, 1.45, c='r')
plt.show()

X = df.iloc[:, 0:4]
y = df.class_value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = KNeighborsClassifier() # (n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

import time

timestamp1 = time.time()
result = []
for k in range(1, 101):
    model = KNeighborsClassifier(k, weights='distance')
    model.fit(X_train, y_train)
    result.append(model.score(X_test, y_test))

timestamp2 = time.time()
print(f'Czas działania pętli - 100 razy: {timestamp2-timestamp1} sekund')
plt.plot(range(1, 101), result)
plt.grid()
plt.show()
