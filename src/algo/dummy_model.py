import logging

from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

def create_features(df_stock, nlags=10):
    df_resampled = df_stock.resample('1D').mean()
    df_resampled = df_resampled[df_resampled.index.to_series().apply(lambda x: x.weekday() not in [5, 6])]
    lags_col_names = []
    for i in range(nlags + 1):
        df_resampled['lags_' + str(i)] = df_resampled['close'].shift(i)
        lags_col_names.append('lags_' + str(i))
    df = df_resampled[lags_col_names]
    df['Y'] = np.where(df['lags_1'] > df['lags_0'], -1, 1)
    df = df.dropna(axis=0)
    return df


def create_X_Y(df_lags):
    X = df_lags.drop(['lags_0', 'Y'], axis=1)
    Y = df_lags['Y']
    return X, Y


class Stock_model(BaseEstimator, ClassifierMixin):

    def __init__(self, data_fetcher):
        self.log = logging.getLogger()
        # self.lr = LinearRegression()
        self.lr = SVC()
        self._data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X, y):
        data = self._data_fetcher(X)
        df_features = create_features(data)
        df_features, Y = create_X_Y(df_features)
        self.lr.fit(df_features, Y)
        return self

    def predict(self, X, Y=None):
        data = self._data_fetcher(X, last=True)
        df_features = create_features(data)
        X, y = create_X_Y(df_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        cls = SVC().fit(X_train, y_train)
        accuracy_train = balanced_accuracy_score(y_train, cls.predict(X_train)) * 100
        accuracy_test = balanced_accuracy_score(y_test, cls.predict(X_test)) * 100
        predictions = cls.predict(X)
        predictions = np.where(predictions == 1, 'Buy', 'Sell')
        predictions = predictions.flatten()[-1]
        # predictions_buy_sell = np.where(predictions = 1, 'Buy', 'Sell')
        # predictions = self.lr.predict(df_features)
        # return predictions.flatten()[-1]
        return predictions, "Train Accuracy:", accuracy_train, "%", "Test Accuracy:", accuracy_test, "%"

