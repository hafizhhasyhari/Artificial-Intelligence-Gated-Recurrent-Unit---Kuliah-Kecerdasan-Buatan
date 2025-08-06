# 📌 Impor pustaka
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense

# 📌 Membaca data
df = pd.read_csv('dataset.csv')  # Ganti dengan path dataset kalian
df['tanggal'] = pd.to_datetime(df['tanggal'])
df.set_index('tanggal', inplace=True)
df = df[['beban_listrik']]

# 📌 Scaling data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# 📌 Membentuk dataset time series
def create_dataset(dataset, time_step=30):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i + time_step])
        y.append(dataset[i + time_step])
    return np.array(X), np.array(y)

time_step = 30
X, y = create_dataset(data_scaled, time_step)

# 📌 Bagi data latih & uji
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 📌 Bentuk input sesuai GRU
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 📌 Bangun model GRU
model = Sequential()
model.add(GRU(50, return_sequences=False, input_shape=(time_step, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 📌 Latih model
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# 📌 Prediksi
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 📌 Kembalikan ke skala asli
train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 📌 Visualisasi hasil
plt.figure(figsize=(14,5))
plt.plot(df.index[-len(y_test):], y_test_actual, label='Aktual')
plt.plot(df.index[-len(test_predict):], test_predict, label='Prediksi GRU')
plt.title('Perbandingan Data Aktual vs Prediksi GRU')
plt.xlabel('Tanggal')
plt.ylabel('Beban Listrik')
plt.legend()
plt.grid()
plt.show()
