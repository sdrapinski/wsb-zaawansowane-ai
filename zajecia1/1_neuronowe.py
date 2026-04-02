import pandas as pd
import matplotlib.pyplot as plt

# jaka jest minimalna siec by to policzic
# minimalna siec to 1 neuron z duza liczba epok (10000+),
# jaka jest optymalna siec by to policzic
# optymalna siec to 3 warstwy, 1 neuron w pierwszej, 8 neuronow w drugiej i 1 neuron w trzeciej
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='rmsprop', loss='mse')

df = pd.read_csv('f-c.csv', usecols=[1, 2])
print(df)

result = model.fit(df.F, df.C, epochs=1000, verbose=2)

C_pred = model.predict(df.F)
plt.scatter(df.F, df.C)
plt.plot(df.F, C_pred, c='r')
plt.show()




