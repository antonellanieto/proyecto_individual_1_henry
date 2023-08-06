import json
import ast 
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor



#lectura del json y creación data frame
rows = []
with open('steam_games.json') as f: 
    for line in f.readlines():
        rows.append(ast.literal_eval(line))

df = pd.DataFrame(rows)


#Limpieza de data

df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
df['genres'] = df['genres'].astype(str)
df['genres'] = df['genres'].str.extract(r"(\w+)")
df = df[df['price'].apply(lambda x: str(x).replace('.', '').isnumeric())]
df['price'] = df['price'].astype(float)
df.dropna(subset=['release_date'], inplace=True)
df['release_date'] = df['release_date'].apply(lambda x: pd.to_datetime(x, errors='coerce'))
df = df.dropna(subset=['release_date'])
df['year'] = df['release_date'].dt.year
df['year'] = df['year'].astype(int)
df.drop('release_date', axis=1, inplace=True)
df.dropna(subset=['metascore'], inplace=True)
df.dropna(subset=['price'], inplace=True)
df.dropna(subset=['early_access'], inplace=True)
df.dropna(subset=['genres'], inplace=True)


columns_to_drop = ['publisher', 'url', 'tags', 'discount_price',
                   'reviews_url', 'id', 'developer', 'title']
#elimino columnas pra el modelo
df_reduced = df.drop(columns_to_drop, axis=1)




df_reduced['genres'] = df_reduced['genres'].str.split(',')
df_reduced = df_reduced.explode('genres')
df_reduced['genres'] = df_reduced['genres'].str.extract(r"(\w+)")


label_encoder = LabelEncoder()
df_reduced['genres_encoded'] = label_encoder.fit_transform(df_reduced['genres'])
df_reduced = df_reduced.groupby(level=0).first()



df_reduced.to_pickle('df_reduced4.pkl')

#Comienzo de el modelo


# Separate the target variable (price) from the features (X)
X = df_reduced.drop(['price', 'sentiment', 'genres', 'app_name', 'specs'], axis=1)
y = df_reduced['price']

# Assuming X and y are defined as before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the gradient boosting regressor model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# Calculate the R2 score
r2 = r2_score(y_test, y_pred)

# print("Root Mean Squared Error (RMSE):", rmse)
# print("R-squared (R2) Score:", r2)

