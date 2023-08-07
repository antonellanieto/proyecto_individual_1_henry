
import numpy as np
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException, Depends, Form, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
from pydantic import BaseModel
from enum import Enum
import pickle
from fastapi.responses import HTMLResponse
from train import model 
from fastapi.staticfiles import StaticFiles
import joblib


app = FastAPI(debug=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

df = pd.read_csv('df_reduced.csv')
df = df.dropna()


#Función para conectar el index, página principal
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Función para obtener los 5 géneros más vendidos en un año
@app.get('/genero/')
def genero(Year: int):
    año_str = str(Year)
     
    #Verificar si el año está disponible en DataFrame
    if año_str not in df['year'].astype(str).unique():
        return "No se encuentra ese año disponible"
    
    # Filtro el DataFrame para el año especificado
    df_year = df[df['year'].astype(str).str.startswith(año_str)]
    
    # Obtener los géneros más vendidos en el año especificado
    top_generos = df_year['genres'].explode().value_counts().head(5).index.tolist()
    return top_generos


# Función para obtener los juegos lanzados en un año
@app.get('/juegos/')
def juegos(Year: int):
    año_str = str(Year)

    #Verificar si el año está disponible en DataFrame
    if año_str not in df['year'].astype(str).unique():
        return "No se encuentra ese año disponible"
    
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['year'].astype(str).str.startswith(año_str)]
    
    # Obtener los juegos lanzados en el año especificado
    juegos_lanzados = df_year['app_name'].tolist()
    return juegos_lanzados



# Función para obtener los 5 specs más repetidos en un año
@app.get('/specs/')
def specs(Year: int):
   
    año_str = str(Year)

    #Verificar si el año está disponible en DataFrame
    if año_str not in df['year'].astype(str).unique():
        return "No se encuentra ese año disponible"
    


    # Filtrar el DataFrame para el año especificado
    df_year = df[df['year'].astype(str).str.startswith(año_str)]
    
    # Obtener los specs más repetidos en el año especificado
    top_specs = df_year['specs'].explode().value_counts().head(5).index.tolist()
    return top_specs



# Función para obtener la cantidad de juegos lanzados en un año con early access
@app.get('/earlyaccess/')
def earlyacces(Year: str):
    año_str = str(Year)
    
    #Verificar si el año está disponible en DataFrame
    if año_str not in df['year'].astype(str).unique():
        return "No se encuentra ese año disponible"
    

    # Filtrar el DataFrame para el año especificado
    mask = (df['year'].astype(str).str.startswith(año_str)) & (df["early_access"] == True)
    df_year = df[mask]
    df_year = df[mask]

    games = len(df_year)
    return {"games": games}


@app.get('/sentiment/')
def sentiment(Year: int):
    año_str = str(Year)

    #Verificar si el año está disponible en DataFrame
    if año_str not in df['year'].astype(str).unique():
        return "No se encuentra ese año disponible"
    
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['year'].astype(str).str.startswith(año_str)]
    
    # Obtener el análisis de sentimiento y contar la cantidad de registros en cada categoría
    analisis_sentimiento = df_year['sentiment'].value_counts().to_dict()
    
    return analisis_sentimiento

@app.get('/metascore/')
def metascore(Year: int):
 
    año_str = str(Year)

    #Verificar si el año está disponible en DataFrame
    if año_str not in df['year'].astype(str).unique():
        return "No se encuentra ese año disponible"
    
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['year'].astype(str).str.startswith(año_str)]
    
    # Obtener los top 5 juegos con mayor metascore en el año especificado
    top_metascore_juegos = df_year.nlargest(5, 'metascore')['app_name'].tolist()
    return top_metascore_juegos



#Carga de modelo
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('trained_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)





label_encoder = joblib.load('label_encoder.pkl')


@app.get('/predict/')
def predict(genre, early_access, metascore, year):

    early_access = early_access.lower() == "true"

    
    # Verificar que el género ingresado esté presente en el LabelEncoder
    if genre not in label_encoder.classes_:
        genres_list = ", ".join(label_encoder.classes_)
        print(f"Error: El género '{genre}' no está presente en el dataset.")
        print(f"Los géneros disponibles son: {genres_list}")
        return None, None
    
    # Obtener el valor codificado del género usando el LabelEncoder
    genre_encoded = label_encoder.transform([genre])[0]
    
    # Verificar que el metascore ingresado esté presente en el dataset
    if float(metascore) not in df["metascore"].unique():
        metascores_list = ", ".join(map(str, df["metascore"].unique()))
        print(f"Error: El metascore '{metascore}' no está presente en el dataset.")
        print(f"Los metascores disponibles son: {metascores_list}")
        return None, None
    
    # Verificar que el año ingresado esté presente en el dataset
    if int(year) not in df["year"].unique():
        min_year = df["year"].min()
        max_year = df["year"].max()
        print(f"Error: El año '{year}' no está presente en el dataset.")
        print(f"El rango de años disponibles es de {min_year} a {max_year}.")
        return None, None
    
    # Crear un DataFrame con las características ingresadas
    data = pd.DataFrame({
        "early_access": [early_access],
        "metascore": [metascore],
        "year": [year],
        "genres_encoded": [genre_encoded],
    })
    
    # Realizar la predicción del precio utilizando el modelo entrenado
    price_pred = model.predict(data)[0]
    
    
   
    return {'Precio': price_pred, "RMSE del modelo": 8.827512792544086}


 
# print(predict('Sports', False, 72, 2013))













# #Prediccion de precios, basado en el modelo entrenado y guardado

# valid_genres = ['Action', 'Strategy', 'Indie', 'Casual', 'Adventure', 'Racing', 'RPG', 'Simulation', 'Massively', 'Sports', 'Free']


# class Genero(Enum):
#     Action = "Action"
#     Strategy = "Strategy"
#     Indie = "Indie"
#     Casual = "Casual"
#     Adventure = "Adventure"
#     Racing = "Racing"
#     RPG = "RPG"
#     Simulation = "Simulation"
#     Massively = "Massively Multiplayer"
#     Sports = "Sports"
#     Free = "Free to Play"





























# @app.get("/predict")
# def predict(metascore: float = None, early_access: bool = None, year: int = None, genero: Genre = None):
#     if metascore is None or early_access is None or year is None or genero is None:
#         raise HTTPException(status_code=400, detail="Missing parameters")

#     # Create input data dictionary
#     input_data = {
#         "metascore": metascore,
#         "early_access": early_access,
#         "year": year,
#     }

#     # Set the selected genre's corresponding encoded column to 1
#     genre_column = f"genres_encoded_{genero.name}"
#     input_data[genre_column] = 1

#     # Create input DataFrame
#     input_df = pd.DataFrame([input_data])

#     try:
#         # Replace 'model.predict' with your actual model prediction code
#         price = model.predict(input_df)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail="Invalid input: " + str(e))

#     return {"price": price[0], "RMSE del modelo": 8.85106790266371}


