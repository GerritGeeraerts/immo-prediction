import csv

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AddPricePerSquareMeter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # Fit method is here for compatibility with the scikit-learn transformer interface.
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['Price Per m²'] = X_transformed['Price'] / X_transformed['Area']
        return X_transformed
