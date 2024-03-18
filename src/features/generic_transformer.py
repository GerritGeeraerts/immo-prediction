import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MyKnnImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column_names: list, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.column_names = column_names
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        # self.X_copy_temp = None

    def fit(self, X, y=None):
        self.imputer.fit(X[self.column_names])
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy_temp = X_copy[self.column_names]
        X_ohe = pd.DataFrame(self.imputer.transform(X_copy_temp), columns=X_copy_temp.columns)
        X_copy.loc[:, self.column_names] = X_ohe
        return X_copy


# My custom min max scaler that reces
class MyMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, multipliers=None):
        self.multipliers = multipliers if multipliers else {}
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        self.column_names = X.columns
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
        X_copy = X.copy()
        if self.keep_columns:
            return X_copy[self.keep_columns]
        X_copy.drop(columns=self.drop_columns, inplace=True)
        return X_copy


#
# class ColumnSelector(BaseEstimator, TransformerMixin):
#     def __init__(self, target_column, keep_columns=None, drop_columns=None):
#         self.keep_columns = keep_columns if keep_columns else []
#         self.drop_columns = drop_columns if drop_columns else []
#         self.target_column = target_column
#         if keep_columns and drop_columns:
#             raise ValueError("You can't have both keep_columns and drop_columns")
#         print('keep_columns:', self.keep_columns)
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X_copy = X.copy()
#         print(self.keep_columns)
#         if self.keep_columns:
#             columns_to_keep = self.keep_columns[:]
#             if self.target_column in X_copy.columns:
#                 columns_to_keep.append(self.target_column)
#             print('columns_to_keep:', columns_to_keep)
#             X_copy = X_copy[columns_to_keep]
#             return X_copy
#
#         columns_to_drop = self.drop_columns[:]
#         if self.target_column in X_copy.columns:
#             columns_to_drop.apped(self.target_column)
#         X_copy = X_copy.drop(columns=columns_to_drop)
#         return X_copy
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
        X_copy = X.copy()
        if self.categories_to_keep:
            X_copy = X_copy[X_copy[self.column].isin(self.categories_to_keep)]
            X_copy = X_copy.reset_index(drop=True)
            return X_copy
        X_copy = X_copy[~X_copy[self.column].isin(self.categories_to_drop)]
        X_copy = X_copy.reset_index(drop=True)
        return X_copy


class MyStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_scale=None):
        self.scaler = StandardScaler()
        self.columns_to_scale = columns_to_scale
        self.columns_not_to_scale = None
        self.all_column_names = None

    def fit(self, df, y=None):
        self.all_column_names = df.columns
        self.columns_not_to_scale = [col for col in self.all_column_names if col not in self.columns_to_scale]
        df_to_scale = self.get_sub_df(df)
        self.columns_to_scale = df_to_scale.columns  # adjust order of columns
        self.scaler.fit(df_to_scale)
        return self

    def get_sub_df(self, df):
        df_to_scale = df
        if len(self.columns_to_scale) > 0:
            df_to_scale = df[self.columns_to_scale]
        return df_to_scale

    def transform(self, df, y=None):
        df_to_scale = self.get_sub_df(df)
        df_scaled = self.scaler.transform(df_to_scale)
        df_result = pd.DataFrame(df_scaled, columns=self.columns_to_scale)
        df_result = pd.concat([df_result, df[self.columns_not_to_scale]], axis=1)
        return df_result


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
