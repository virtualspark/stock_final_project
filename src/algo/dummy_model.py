import logging

from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

def create_features(df_stock, nlags=1):
    df_resampled = df_stock.resample('1D').mean()
    df_resampled = df_resampled[df_resampled.index.to_series().apply(lambda x: x.weekday() not in [5, 6])]

    # df_resampled = df_resampled.reset_index()
    # df_resampled = df_resampled.where(df_resampled['index'] >= "2020-01-01")
    # df_resampled = df_resampled.where(df_resampled)
    # lags_col_names = []
    # for i in range(nlags + 1):
    #     df_resampled['lags_' + str(i)] = df_resampled['close'].shift(i)
    #     lags_col_names.append('lags_' + str(i))

    df_resampled['open_close'] = df_resampled.open - df_resampled.close
    df_resampled['high_low'] = df_resampled.high - df_resampled.low
    df_resampled['D+1'] = df_resampled['close'].shift(0)
    df_resampled['D'] = df_resampled['close'].shift(1)

    def normalize(x, col_max):
        if x == -1:
            return np.nan
        else:
            return x / col_max
    df_resampled['volume'] = df_resampled['volume'].apply(lambda x: normalize(x, df_resampled['volume'].max()))
    df_resampled['open_close'] = df_resampled['open_close'].apply(lambda x: normalize(x, df_resampled['open_close'].max()))
    df_resampled['high_low'] = df_resampled['high_low'].apply(lambda x: normalize(x, df_resampled['high_low'].max()))

    df = df_resampled[['D+1', 'D', 'open_close', 'high_low', 'volume']]

    df['Y'] = np.where(df['D'] > df['D+1'], -1, 1)
    df = df.dropna(axis=0)
    print(df)
    return df


def create_X_Y(df_lags):
    X = df_lags.drop(['Y', 'D+1', 'D'], axis=1)
    Y = df_lags['Y']
    return X, Y


class Stock_model(BaseEstimator, ClassifierMixin):

    def __init__(self, data_fetcher):
        self.log = logging.getLogger()
        # self.lr = LinearRegression()
        # self.lr = SVC()
        self.lr = LogisticRegression()
        self._data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X, y=None):
        data = self._data_fetcher(X)
        df_features = create_features(data)
        df_features, Y = create_X_Y(df_features)
        self.lr.fit(df_features, Y)
        return self

    def predict(self, X, Y=None):
        data = self._data_fetcher(X, last=True)
        df_features = create_features(data)
        X, y = create_X_Y(df_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
        # cls = SVC().fit(X_train, y_train)
        cls = LogisticRegression(random_state=0).fit(X_train, y_train)
        accuracy_train = balanced_accuracy_score(y_train, cls.predict(X_train)) * 100
        accuracy_test = balanced_accuracy_score(y_test, cls.predict(X_test)) * 100
        predictions = cls.predict(X)
        predictions = np.where(predictions == 1, 'Buy', 'Sell')
        predictions = predictions.flatten()[-1]
        # predictions_buy_sell = np.where(predictions = 1, 'Buy', 'Sell')
        # predictions = self.lr.predict(df_features)
        # return predictions.flatten()[-1]
        return predictions, "Train Accuracy:", accuracy_train, "%", "Test Accuracy:", accuracy_test, "%"

