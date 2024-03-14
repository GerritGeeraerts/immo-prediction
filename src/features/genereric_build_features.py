import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class OneHotEncodeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.one_hot_encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            ohe.fit(X[[column]])
            self.one_hot_encoders[column] = ohe
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column, ohe in self.one_hot_encoders.items():
            X_ohe = pd.DataFrame(
                ohe.transform(X_copy[[column]]),
                columns=ohe.get_feature_names_out(input_features=[column]),
                index=X_copy.index

            )
            X_copy = pd.concat([X_copy.drop(columns=[column]), X_ohe], axis=1)
        return X_copy
