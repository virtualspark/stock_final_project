import logging

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

def create_features(df_stock, nlags=1):
    df_resampled = df_stock.resample('1D').mean()
    df_resampled = df_resampled[df_resampled.index.to_series().apply(lambda x: x.weekday() not in [5, 6])]
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

    def predict(self, X, y=None):
        data = self._data_fetcher(X, last=True)
        df_features = create_features(data)
        X, y = create_X_Y(df_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)
        clf = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=10, bootstrap=True).fit(X, y)
        accuracy_train_clf = balanced_accuracy_score(y_train, clf.predict(X_train)) * 100
        accuracy_test_clf = balanced_accuracy_score(y_test, clf.predict(X_test)) * 100
        predictions_clf = clf.predict(X)
        predictions_clf = np.where(predictions_clf == 1, 'Buy', 'Sell')
        predictions_clf = predictions_clf.flatten()[-1]
        print(accuracy_train_clf, accuracy_test_clf)
        return predictions_clf

