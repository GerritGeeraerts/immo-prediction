import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from features.generic_transformer import MyStandardScaler
from features.genereric_build_features import OneHotEncodeColumns, PostalCodePimp
from utils import save_model_as_pickle

df = pd.read_csv('/home/gg/PycharmProjects/immo-prediction/data/raw/data.csv', low_memory=False)
df.head()

sub_types_to_keep = [
    'VILLA', 'HOUSE', 'APARTMENT',
]
columns_to_keep = ['Bathroom Count', 'Bedroom Count', 'Habitable Surface', 'Land Surface', 'Price', 'Subtype', 'Postal Code',]

# Fix some data in the dataframe
df.loc[df['Subtype'] == 'APARTMENT', 'Land Surface'] = 0
df = df.dropna(subset=columns_to_keep)
# df = df[:10000]
df = df.reset_index(drop=True)
df = df[df['Subtype'].isin(sub_types_to_keep)]
df = df[columns_to_keep]
df['Postal Code'] = df['Postal Code'].apply(lambda x: int(str(x)[:2]))
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

# postal code pimp
postal_pimp = PostalCodePimp()
postal_pimp.fit(training_data)
training_data = postal_pimp.transform(training_data)
testing_data = postal_pimp.transform(testing_data)
print(testing_data.isna().sum())
# impute missing values with mean for colum postal code
print(testing_data.isna().sum())

print(training_data)


# one hot encode the subtype
OneHotEncodeColumns = OneHotEncodeColumns(['Subtype'])
training_data = OneHotEncodeColumns.fit_transform(training_data)
testing_data = OneHotEncodeColumns.transform(testing_data)

# standard scaler
standard_scaler = MyStandardScaler(columns_to_scale=['Habitable Surface', 'Land Surface'])

# training data
training_columns = training_data.columns
X_training = standard_scaler.fit_transform(training_data)
training_data = pd.DataFrame(X_training, columns=training_columns)

# testing data
testing_columns = testing_data.columns
testing_data = standard_scaler.transform(testing_data)
testing_data = pd.DataFrame(testing_data, columns=testing_columns)


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

save_model_as_pickle(reg_model, '../../models/basic_model_with_subtype_postal_code.pkl')
