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


class PostalCodePimp(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.postal_code_price_sqm = {}
        self.avg_price = 0

    def fit(self, df, y=None):
        postal_code_price_sqm = {}
        df_copy = df.copy()
        df_copy["price_per_sqm"] = df_copy['Price'] / df['Habitable Surface']
        df_copy["Postal Value"] = df_copy.groupby('Postal Code')["price_per_sqm"].transform('mean')
        self.avg_price = df_copy["Postal Value"].mean()
        df_copy = df_copy[['Postal Code', 'Postal Value']].sort_values('Postal Code').drop_duplicates()
        df_copy.set_index('Postal Code', inplace=True)
        self.postal_code_price_sqm = df_copy.to_dict()
        return self

    def transform(self, df):
        df_copy = df.copy()
        df_copy['Postal Code'] = df_copy['Postal Code'].map(self.postal_code_price_sqm['Postal Value'])
        df_copy.fillna({'Postal Code': self.avg_price}, inplace=True)
        return df_copy

class BuildingStatePimp(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None):
        postal_code_price_sqm = {}
        df_copy = df.copy()
        df_copy["price_per_sqm"] = df_copy['Price'] / df['Habitable Surface']
        df_copy["bsp"] = df_copy.groupby('State of Building')["price_per_sqm"].transform('mean')
        breakpoint()
        return self

    def transform(self, df):
        df_copy = df.copy()
        df_copy['Postal Code'] = df_copy['Postal Code'].map(self.postal_code_price_sqm['Postal Value'])
        df_copy.fillna({'Postal Code': self.avg_price}, inplace=True)
        return df_copy