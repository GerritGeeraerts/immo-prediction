import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, PolynomialFeatures

from features.generic_transformer import MyStandardScaler
from features.genereric_build_features import OneHotEncodeColumns, PostalCodePimp
from utils import save_model_as_pickle

df = pd.read_csv('/home/gg/PycharmProjects/immo-prediction/data/raw/data.csv', low_memory=False)
df.head()
print(df.shape)

sub_types_to_keep = [
    'VILLA', 'HOUSE', 'APARTMENT',
]
columns_to_keep = ['Bathroom Count', 'Bedroom Count', 'Habitable Surface', 'Land Surface', 'Price', 'Subtype',
                   'Latitude', 'Longitude',]

# Fix some data in the dataframe
df.loc[df['Subtype'] == 'APARTMENT', 'Land Surface'] = 0
df = df.dropna(subset=['Bathroom Count', 'Bedroom Count', 'Habitable Surface', 'Subtype', 'Latitude', 'Longitude', ])
print(df.shape)
# df = df[:50000)
df = df.reset_index(drop=True)
df = df[df['Subtype'].isin(sub_types_to_keep)]
print(df.shape)
df = df[columns_to_keep]
df.reset_index(drop=True, inplace=True)
# epc_map = {
#     "A": 7,
#     "B": 6,
#     "C": 5,
#     "D": 4,
#     "E": 3,
#     "F": 2,
#     "G": 1,
# }
#
#
# def replace_value(x):
#     for k, v in epc_map.items():
#         if str(k) in str(x):
#             return v
#     return -1


# the count of each value in EPC column
# df['EPC'] = df['EPC'].apply(replace_value)
# kitchen_map = {
#     "INSTALLED": 1,
#     "HYPER_EQUIPPED": 3,
#     "SEMI_EQUIPPED": 2,
#     "USA_HYPER_EQUIPPED": 3,
#     "NOT_INSTALLED": 0,
#     "USA_INSTALLED": 1,
#     "USA_SEMI_EQUIPPED": 2,
#     "USA_UNINSTALLED": 0,
# }
# df['Kitchen Type'] = df['Kitchen Type'].map(lambda x: kitchen_map.get(x, -1))
# state_map = {
#     "AS_NEW": 5,
#     "JUST_RENOVATED": 4,
#     "GOOD": 3,
#     "TO_BE_DONE_UP": 2,
#     "TO_RENOVATE": 1,
#     "TO_RESTORE": 0,
# }
# df['State of Building'] = df['State of Building'].map(lambda x: state_map.get(x, -1))
df.reset_index(drop=True, inplace=True)

X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)

# training data re-add the price column
training_data = pd.concat([X_train, y_train], axis=1)
training_data.reset_index(drop=True, inplace=True)

# testing data re-add the price column with np.nan
testing_data = X_test.copy()
testing_data['Price'] = np.nan
testing_data.reset_index(drop=True, inplace=True)

ordinal_enc = OrdinalEncoder()
training_data['Subtype'] = ordinal_enc.fit_transform(training_data[['Subtype']]) * 100
testing_data['Subtype'] = ordinal_enc.transform(testing_data[['Subtype']]) * 100

min_max_scaler_col = ['Longitude', 'Latitude',]
min_max_scaler = MinMaxScaler()
training_data[min_max_scaler_col] = min_max_scaler.fit_transform(
    training_data[min_max_scaler_col])
testing_data[min_max_scaler_col] = min_max_scaler.transform(
    testing_data[min_max_scaler_col])

# training_data['Building state price'] = training_data['Price'] / training_data['Habitable Surface']
# testing_data['Building state price'] = np.nan
#
# knn_bs_columns = ['Subtype', 'State of Building', 'EPC', 'Kitchen Type', 'Building state price', 'Latitude', 'Longitude']
# knn_bs = KNNImputer(n_neighbors=50)
# knn_bs.fit(training_data[knn_bs_columns])
# result_bs = knn_bs.transform(testing_data[knn_bs_columns])
# result_bs = pd.DataFrame(result_bs, columns=knn_bs_columns)
# testing_data['Building state price'] = result_bs['Building state price']
# columns_to_keep.append('Building state price')


training_data['Locality Typed Price'] = training_data['Price'] / training_data['Habitable Surface']
# in testing data create a new column with np.nan named 'Locality Typed Price'
testing_data['Locality Typed Price'] = np.nan
# print(training_data)
# print(testing_data)

# use KNN to find the closest 5 neighbours
knn_columns = ['Longitude', 'Latitude', 'Subtype', 'Locality Typed Price']
knn = KNNImputer(n_neighbors=25)
knn.fit(training_data[knn_columns])
result = knn.transform(testing_data[knn_columns])
# create a new dataframe with the result
result = pd.DataFrame(result, columns=knn_columns)
# concatenate the result with the testing data
testing_data['Locality Typed Price'] = result['Locality Typed Price']
print(testing_data)
columns_to_keep.append('Locality Typed Price')

# MyOneHotEncodeColumns = OneHotEncodeColumns(['State of Building', 'EPC', 'Kitchen Type'])
# training_data = MyOneHotEncodeColumns.fit_transform(training_data)
# testing_data = MyOneHotEncodeColumns.transform(testing_data)

# training_data.drop(columns=['Subtype'], inplace=True)
# testing_data.drop(columns=['Subtype'], inplace=True)

# one hot encode the subtype
# OneHotEncodeColumns = OneHotEncodeColumns(['Subtype'])
# training_data = OneHotEncodeColumns.fit_transform(training_data)
# testing_data = OneHotEncodeColumns.transform(testing_data)


standard_scaler = MyStandardScaler(columns_to_scale=['Habitable Surface', 'Land Surface', 'Locality Typed Price', ])

# training data
training_columns = training_data.columns
X_training = standard_scaler.fit_transform(training_data)
training_data = pd.DataFrame(X_training, columns=training_columns)

# testing data
testing_columns = testing_data.columns
testing_data = standard_scaler.transform(testing_data)
testing_data = pd.DataFrame(testing_data, columns=testing_columns)  # error

# get the X_train and y_train
X_train = training_data.drop(columns=['Price'])
y_train = training_data['Price']

# get the X_test and y_test
X_test = testing_data.drop(columns=['Price'])

poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X_train)
print(X_poly.shape)
print(X_poly)
print(X_test.shape)
print(X_test)
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_poly, y_train)
X_test_poly = poly.transform(X_test)
y_pred = poly_reg_model.predict(X_test_poly)
print(y_pred.shape)
# reg_model = LinearRegression()
#
# reg_model.fit(X_train, y_train)
#
# print(reg_model.score(X_test, y_test))
#
# # 5. Predict the target values for the testing data
# y_pred = reg_model.predict(X_test)

# 6. Calculate the MSE
print(len(y_test))
print(len(y_pred))
mse = mean_squared_error(y_test, y_pred)


# Calculate the R-squared value
r_squared = r2_score(y_test, y_pred)
print(f'R-squared value: {r_squared:.2%}')
