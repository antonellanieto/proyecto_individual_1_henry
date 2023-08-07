
import numpy as np
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from pydantic import BaseModel
from enum import Enum
import pickle
from fastapi.responses import HTMLResponse
from train import model 
from fastapi.staticfiles import StaticFiles



app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
df = pd.read_csv('df_reduced.csv')



#Función para conectar el index, página principal
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r", encoding="utf-8") as file:
        content = file.read()
    return HTMLResponse(content=content)


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


#Prediccion de precios, basado en el modelo entrenado y guardado



# Crear el Enum de géneros
class Genre(Enum):
    Action = "Action"
    Adventure = "Adventure"
    Casual = "Casual"
    Early_Access = "Early Access"
    Free_to_Play = "Free to Play"
    Indie = "Indie"
    Massively_Multiplayer = "Massively Multiplayer"
    RPG = "RPG"
    Racing = "Racing"
    Simulation = "Simulation"
    Sports = "Sports"
    Strategy = "Strategy"
    Video_Production = "Video Production"

# Definir la ruta de predicción
@app.get("/predict") 
def predict(metascore: float = None, earlyaccess: bool = None, Año: str = None, genero: Genre = None):
    # Validar que se hayan pasado los parámetros necesarios
    if metascore is None or Año is None or genero is None or earlyaccess is None:
        raise HTTPException(status_code=400, detail="Missing parameters: metascore, year, genero, and earlyaccess are required.")
    
    # Convertir el input en un DataFrame con las columnas necesarias para el modelo
    input_df = pd.DataFrame([[metascore, earlyaccess, Año, *[1 if genero.value == g else 0 for g in Genre._member_names_]]], columns=['metascore', 'earlyaccess', 'year', *Genre._member_names_])
    
    # Verificar si el género es Free to Play
    if genero == Genre.Free_to_Play:
        # Devolver 0 como precio
        return {"price": 0, "RMSE del modelo": 8.36}
    else:
        # Realizar la predicción con el modelo
        try:
            # Replace 'model.predict' with your actual model prediction code
            price = model.predict(input_df)[0]
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid input: " + str(e))

        # Devolver el precio y el RMSE como salida
        return {"price": price, "RMSE del modelo": 8.83}