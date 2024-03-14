import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MyKnnImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column_names: list, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.column_names = column_names
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        return pd.DataFrame(self.imputer.transform(X_copy), columns=X.columns)


# My custom min max scaler that reces
class MyMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, multipliers=None):
        self.multipliers = multipliers if multipliers else {}
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy[self.column_names] = self.scaler.transform(X_copy[self.column_names])
        for column, multiplier in self.multipliers.items():
            X_copy[column] = X_copy[column] * multiplier

        return X_copy


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, keep_columns=None, drop_columns=None):
        self.keep_columns = keep_columns if keep_columns else []
        self.drop_columns = drop_columns if drop_columns else []
        if keep_columns and drop_columns:
            raise ValueError("You can't have both keep_columns and drop_columns")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.keep_columns] if self.keep_columns else X.drop(columns=self.drop_columns)


class RowSelectorByCategory(BaseEstimator, TransformerMixin):
    def __init__(self, column, categories_to_keep=None, categories_to_drop=None):
        if categories_to_keep and categories_to_drop:
            raise ValueError("You can't have both categories_to_keep and categories_to_drop")
        self.all_categories = None
        self.column = column
        self.categories_to_keep = categories_to_keep if categories_to_keep else []
        self.categories_to_drop = categories_to_drop if categories_to_drop else []

    def fit(self, X, y=None):
        self.all_categories = X[self.column].unique()
        return self

    def transform(self, X):
        if self.categories_to_keep:
            return X[X[self.column].isin(self.categories_to_keep)]
        return X[~X[self.column].isin(self.categories_to_drop)]


class DataFrameStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns = None  # To store the column names for transformation

    def fit(self, X, y=None):
        self.scaler.fit(X)
        self.columns = X.columns  # Store the column names from fitting
        return self

    def transform(self, X, y=None):
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, index=X.index, columns=self.columns)
