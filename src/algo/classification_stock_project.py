import logging

# classification model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn
plt.interactive(False)

import yfinance as yf

df = yf.download('AAPL', start="2015-01-01", end="2020-10-14")
df = df.dropna()
df.Close.plot(figsize=(10,5))
plt.ylabel("S&P500 Price")
plt.show()

# Creating the D and D+1 columns
df['D'] = df['Close'].shift(1)
df['D+1'] = df['Close']

# Creating the output to get the signal: sell (-1) and buy (1)
y = np.where(df['D'] > df['D+1'], -1, 1)

# Creating my predictors for the model
X = df[['Open', 'Close', 'High', 'Low', 'Adj Close', 'Volume']]

# Splitting training datasets and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Creating the model (SVC model)
cls = SVC().fit(X_train, y_train)

# Model Evaluation using accuracy
accuracy_train = accuracy_score(y_train, cls.predict(X_train))
accuracy_test = accuracy_score(y_test, cls.predict(X_test))

print('Train Accuracy:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))

df['Predicted_Signal'] = cls.predict(X)

print(df)