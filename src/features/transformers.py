import re

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

from config import TYPES_TO_KEEP, SUBTYPES_TO_KEEP
from utils import ImmoFeature as IF


# class OutlierRemover(BaseEstimator, TransformerMixin):
#     def __init__(self, factor=1.5):
#         self.factor = factor
#
#     def outlier_removal(self, X, y=None):
#         pass
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         return X.apply(self.outlier_removal)


class PostalCodeFixer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Replaces all non-digit chars in the postal code column
        """
        X_copy = X.copy()
        X_copy[IF.postal_code.value] = X_copy[IF.postal_code.value].apply(lambda x: re.sub('[^\d]', '', str(x)))
        return X_copy


class PropertyTypeDropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Drops rows with missing values for Immo
        """
        X_copy = X.copy()

        X_filtered = X_copy[
            X_copy[IF.type.value].isin(TYPES_TO_KEEP) &
            X_copy[IF.subtype.value].isin(SUBTYPES_TO_KEEP) &
            X_copy[IF.sale_type.value].isin(['NORMAL_SALE'])
            ]

        # reset index
        X_filtered.reset_index(drop=True, inplace=True)
        return X_filtered


class CrucialPropertiesMissingDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Drops rows with missing values for price, Postal Code, Habitable Surface, bedroom count
        """
        X_copy = X.copy()

        required_columns = [IF.price.value, IF.postal_code.value, IF.habitable_surface.value, IF.bedroom_count.value]

        X_filtered = X_copy.dropna(subset=required_columns)
        # reset index
        X_filtered.reset_index(drop=True, inplace=True)
        return X_filtered


class LandSurfaceFixer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        For all appartments, set land_surface to 0
        """
        X_copy = X.copy()
        columns_to_set_to_zero = [
            'VILLA', 'HOUSE', 'APARTMENT', 'MANSION', 'PENTHOUSE', 'TOWN_HOUSE', 'GROUND_FLOOR', 'FLAT_STUDIO',
            'DUPLEX',
        ]
        X_copy.loc[X_copy[IF.apartment.value] == 1, IF.land_surface.value] = 0
        X_copy.loc[X_copy[IF.penthouse.value] == 1, IF.land_surface.value] = X_copy.loc[
            X_copy[IF.penthouse.value] == 1, IF.land_surface.value].fillna(0)
        X_copy.loc[X_copy[IF.ground_floor.value] == 1, IF.land_surface.value] = X_copy.loc[
            X_copy[IF.ground_floor.value] == 1, IF.land_surface.value].fillna(0)
        X_copy.loc[X_copy[IF.flat_studio.value] == 1, IF.land_surface.value] = X_copy.loc[
            X_copy[IF.flat_studio.value] == 1, IF.land_surface.value].fillna(0)
        X_copy.loc[X_copy[IF.duplex.value] == 1, IF.land_surface.value] = X_copy.loc[
            X_copy[IF.duplex.value] == 1, IF.land_surface.value].fillna(0)

        return X_copy


class ColumnDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Drops columns with more than threshold missing values
        """
        X_copy = X.copy()
        columns_to_drop = [
            IF.garden_surface.value,  # insufficient values
            IF.parking_count_inside.value,  # insufficient values
            IF.parking_count_outside.value,  # insufficient values
            IF.living_surface.value,  # insufficient values
            IF.kitchen_surface.value,  # insufficient values
            IF.sea_view.value,  # insufficient values
            IF.is_holiday_property.value,  # insufficient values
            IF.terrace_orientation.value,  # insufficient values
            IF.fireplace_count.value,  # insufficient values TODO test to complete missing with 0
            IF.garden_orientation.value,  # insufficient values
            IF.sewer.value,  # insufficient values TODO test to complete missing with 0
            IF.gas_water_electricity.value,  # insufficient values TODO test to complete missing with 1
            IF.swimming_pool.value,  # insufficient values TODO test to complete missing with 0
            IF.id.value,  # low correlation
            IF.room_count.value,  # low correlation
            IF.toilet_count.value,  # low correlation
            IF.cadastral_income.value,  # low correlation
            IF.build_year.value,  # low correlation
            IF.furnished.value,  # irrelevant
            IF.openfire.value,  # irrelevant
            IF.terrace.value,  # irrelevant
            IF.garden_exists.value,  # irrelevant
            IF.has_starting_price.value,  # irrelevant
            IF.transaction_subtype.value,  # irrelevant
            IF.url.value,  # irrelevant
            IF.sale_type.value,  # irrelevant
            IF.kitchen.value,  # irrelevant
            IF.type.value,  # irrelevant
            IF.locality.value,  # irrelevant
        ]
        X_filtered = X_copy.drop(columns=columns_to_drop)
        return X_filtered


class TerraceSurfaceFixer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[IF.terrace_surface.value] = X_copy[IF.terrace_surface.value].fillna(0)
        return X_copy


class MyMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, excluded=None):
        if not excluded:
            excluded = []
        self.exclude = excluded

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Min-Max scaling on all columns except the excluded ones
        columns_to_scale = [column for column in X_copy.columns if column not in self.exclude]

        X_copy_to_scale = X_copy[columns_to_scale]
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(X_copy_to_scale), columns=X_copy_to_scale.columns)
        df_scaled = pd.concat([X_copy[self.exclude], df_scaled], axis=1)
        # for column in X_copy.columns:
        #     if column not in self.exclude:
        #         X_copy[column] = (X_copy[column] - X_copy[column].min()) / (X_copy[column].max() - X_copy[column].min())
        return df_scaled


class FacadeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[IF.facades.value] = X_copy[IF.facades.value].apply(lambda x: 2 if x < 2 else x)
        X_copy[IF.facades.value] = X_copy[IF.facades.value].apply(lambda x: 4 if x > 4 else x)

        # fill villa with 4 facades
        X_copy.loc[X_copy[IF.villa.value] == 1, IF.facades.value] = X_copy.loc[
            X_copy[IF.villa.value] == 1, IF.facades.value].fillna(4)

        temp_df = X_copy[[IF.facades.value, 'Price_scaled', IF.bathroom_count.value, IF.bedroom_count.value,
                          IF.villa.value]]
        imputer = KNNImputer(n_neighbors=5)
        temp_df_imputed = pd.DataFrame(imputer.fit_transform(temp_df), columns=temp_df.columns)
        X_copy[IF.facades.value] = temp_df_imputed[IF.facades.value]
        return X_copy


class LongitudeLatitudeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df_copy = df.copy()

        # Step 1: Isolate postal_code and longitude columns
        temp_df = df_copy[[IF.postal_code.value, IF.longitude.value]]

        # Step 2: Apply KNN Imputer
        imputer = KNNImputer(n_neighbors=1)
        temp_df_imputed = pd.DataFrame(imputer.fit_transform(temp_df), columns=temp_df.columns)

        # Step 3: Update the original DataFrame
        df_copy[IF.longitude.value] = temp_df_imputed[IF.longitude.value]

        temp_df = df_copy[[IF.postal_code.value, IF.latitude.value]]

        # Step 2: Apply KNN Imputer
        imputer = KNNImputer(n_neighbors=1)
        temp_df_imputed = pd.DataFrame(imputer.fit_transform(temp_df), columns=temp_df.columns)

        # Step 3: Update the original DataFrame
        df_copy[IF.latitude.value] = temp_df_imputed[IF.latitude.value]

        return df_copy


class LandSurfaceImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[IF.land_surface.value] = X_copy[IF.land_surface.value].fillna(0)
        return X_copy


class BathroomCountImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df_copy = df.copy()

        # Step 1: Create a temporary DataFrame with relevant columns
        temp_df = df_copy[[
            IF.bathroom_count.value,
            IF.apartment.value, IF.house.value, IF.villa.value, IF.mansion.value, IF.penthouse.value,
            IF.town_house.value, IF.ground_floor.value, IF.flat_studio.value, IF.duplex.value, 'Price_scaled',
            IF.habitable_surface.value, IF.bedroom_count.value
        ]]

        # Step 2: Apply KNN Imputer
        imputer = KNNImputer(n_neighbors=5)  # Adjust n_neighbors as needed
        temp_df_imputed = pd.DataFrame(imputer.fit_transform(temp_df), columns=temp_df.columns)

        # Step 3: Update the original DataFrame
        df_copy[IF.bathroom_count.value] = temp_df_imputed[IF.bathroom_count.value]
        return df_copy


class MyKnnImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target, columns, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.columns = [target].extend(columns)
        self.target = target

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df_copy = df.copy()
        temp_df = df_copy[[self.columns]]

        imputer = KNNImputer(n_neighbors=self.n_neighbors)

        temp_df_imputed = pd.DataFrame(imputer.fit_transform(temp_df), columns=temp_df.columns)
        # Step 1: Create a temporary DataFrame with relevant columns
        df_copy[self.target] = temp_df_imputed[self.target]


class UnderpopulatedBinaryColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.04):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # for each column get the distinct values. If the distinct values are 0 and 1 put the column in a list
        binary_columns = []
        for column in X_copy.columns:
            if X_copy[column].nunique() == 2 and 0 in X_copy[column].unique() and 1 in X_copy[column].unique():
                binary_columns.append(column)

        X_binary = X_copy[binary_columns]

        # count the number of 1s in each column and divide by the number
        # of rows if the result is less than the threshold drop the column
        for column in X_binary.columns:
            if X_binary[column].sum() / X_binary.shape[0] < self.threshold:
                X_copy = X_copy.drop(columns=[column])
        return X_copy


class ColumnKeeper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

# class BinaryToFloatTransformer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X_copy = X.copy()
#         X_copy = X_copy[IF.kitchen.value].fillna(False)
#         X_copy[IF.kitchen.value] = X_copy[IF.kitchen.value].apply(lambda x: float(x))
#         return X_copy
