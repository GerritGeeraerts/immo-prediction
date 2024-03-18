import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

from features.generic_transformer import MyStandardScaler
from features.genereric_build_features import OneHotEncodeColumns, PostalCodePimp
from utils import save_model_as_pickle

df = pd.read_csv('/home/gg/PycharmProjects/immo-prediction/data/raw/data.csv', low_memory=False)
df.head()

sub_types_to_keep = [
    'VILLA', 'HOUSE', 'APARTMENT',
]
columns_to_keep = ['Bathroom Count', 'Bedroom Count', 'Habitable Surface', 'Land Surface', 'Price', 'Subtype',
                   'Latitude', 'Longitude', ]

# Fix some data in the dataframe
df.loc[df['Subtype'] == 'APARTMENT', 'Land Surface'] = 0
df = df.dropna(subset=columns_to_keep)
# drop properties with a habitable surface greater than 500
df = df[df['Habitable Surface'] < 1000]
# df = df[:50000]
df = df.reset_index(drop=True)
df = df[df['Subtype'].isin(sub_types_to_keep)]
df = df[columns_to_keep]
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

training_data['Locality Typed Price'] = training_data['Price'] / training_data['Habitable Surface']
# in testing data create a new column with np.nan named 'Locality Typed Price'
testing_data['Locality Typed Price'] = np.nan
print(training_data)
print(testing_data)


# print max and min of the columns
print(training_data[['Longitude', 'Latitude', 'Habitable Surface']].max())

min_max_scaler = MinMaxScaler()
fields = ['Longitude', 'Latitude', 'Habitable Surface']
training_data[fields] = min_max_scaler.fit_transform(training_data[fields])
testing_data[fields] = min_max_scaler.transform(testing_data[fields])

# use KNN to find the closest 5 neighbours
knn = KNNImputer(n_neighbors=25)
knn_fields = ['Longitude', 'Latitude', 'Subtype', 'Locality Typed Price']
knn.fit(training_data[knn_fields])
result = knn.transform(testing_data[knn_fields])
# create a new dataframe with the result
result = pd.DataFrame(result, columns=knn_fields)
# concatenate the result with the testing data
testing_data['Locality Typed Price'] = result['Locality Typed Price']
print(testing_data)
columns_to_keep.append('Locality Typed Price')

# training_data.drop(columns=['Subtype'], inplace=True)
# testing_data.drop(columns=['Subtype'], inplace=True)

# one hot encode the subtype
# OneHotEncodeColumns = OneHotEncodeColumns(['Subtype'])
# training_data = OneHotEncodeColumns.fit_transform(training_data)
# testing_data = OneHotEncodeColumns.transform(testing_data)

# standard scaler
# standard_scaler = MyStandardScaler(columns_to_scale=['Habitable Surface', 'Land Surface'])
#
# # training data
# training_columns = training_data.columns
# X_training = standard_scaler.fit_transform(training_data)
# training_data = pd.DataFrame(X_training, columns=training_columns)
#
# # testing data
# testing_columns = testing_data.columns
# testing_data = standard_scaler.transform(testing_data)
# testing_data = pd.DataFrame(testing_data, columns=testing_columns) # error

# get the X_train and y_train
X_train = training_data.drop(columns=['Price'])
y_train = training_data['Price']

# get the X_test and y_test
X_test = testing_data.drop(columns=['Price'])

reg_model = LinearRegression()

reg_model.fit(X_train, y_train)

print(reg_model.score(X_test, y_test))

# 5. Predict the target values for the testing data
y_pred = reg_model.predict(X_test)

# 6. Calculate the MSE
mse = mean_squared_error(y_test, y_pred)

# Calculate the R-squared value
r_squared = r2_score(y_test, y_pred)
print(f'R-squared value: {r_squared:.2%}')

save_model_as_pickle(reg_model, '../../models/train_neigbours.pkl')
