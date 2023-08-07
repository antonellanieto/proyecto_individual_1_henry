import json
import ast 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,RepeatedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

df = pd.read_csv('df_steam.csv')



from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

columns_for_X = ['early_access', 'metascore', 'genres_encoded', 'year']
X = df[columns_for_X]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Bagging Regressor model with DecisionTreeRegressor as base estimator
base_estimator = DecisionTreeRegressor(random_state=42)
model = BaggingRegressor(base_estimator=base_estimator, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)









