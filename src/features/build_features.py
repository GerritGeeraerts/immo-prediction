from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from config import SUBTYPES_TO_KEEP
from utils import ImmoFeature as IF
from .transformers import PropertyTypeDropper


class SubtypeOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')

        X_ohe = ohe.fit_transform(X_copy[[IF.subtype.value]])
        X_copy_with_ohe = pd.concat([X_copy, X_ohe], axis=1).drop(columns=[IF.subtype.value])

        return X_copy_with_ohe


class OneHotEncodeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        for column in self.columns:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')

            X_ohe = ohe.fit_transform(X_copy[[column]])
            X_copy = pd.concat([X_copy, X_ohe], axis=1).drop(columns=[column]).copy()

        return X_copy


class CopyColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column, new_column_name):
        self.column = column
        self.new_column_name = new_column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.new_column_name] = X_copy[self.column]
        return X_copy


class AveragePricePerSqm(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.column] = X_copy[IF.price.value] / X_copy[IF.habitable_surface.value]
        return X_copy


class PostalCodePimp(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """replace the postal code by the mean average price per sqm of the postal code"""
        X_copy = X.copy()

        # calculate the price per sqm
        X_copy["price_per_sqm"] = X_copy[IF.price.value] / X_copy[IF.habitable_surface.value]

        # calculate the mean price per sqm per postal code
        X_copy["mean_price_per_sqm_per_postal_code"] = X_copy.groupby(IF.postal_code.value)["price_per_sqm"].transform('mean')

        # replace the price per sqm by the mean price per sqm per postal code
        X_copy[IF.postal_code.value] = X_copy["mean_price_per_sqm_per_postal_code"]

        return X_copy
