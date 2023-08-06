import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from enum import Enum
import numpy as np

#Comienzo de la api
#para levantar fast api: uvicorn main:app --reload
app = FastAPI()

df = pd.read_csv('df_reduced.csv')

@app.get("/")
async def root():
    return 'Hello, World'


# Función para obtener los 5 géneros más vendidos en un año
@app.get('/genero/')
def genero(Año: int):
    año_str = str(Año)

#     # Filtrar el DataFrame para el año especificado
    df_year = df[df['year'].astype(str).str.startswith(año_str)]
    
    # Obtener los géneros más vendidos en el año especificado
    top_generos = df_year['genres'].explode().value_counts().head(5).index.tolist()
    return top_generos


# Función para obtener los juegos lanzados en un año
@app.get('/juegos/')
def juegos(Año: int):
    año_str = str(Año)

    # Filtrar el DataFrame para el año especificado
    df_year = df[df['year'].astype(str).str.startswith(año_str)]
    
    # Obtener los juegos lanzados en el año especificado
    juegos_lanzados = df_year['app_name'].tolist()
    return juegos_lanzados



# Función para obtener los 5 specs más repetidos en un año
@app.get('/specs/')
def specs(Año: int):

    año_str = str(Año)
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['year'].astype(str).str.startswith(año_str)]
    
    # Obtener los specs más repetidos en el año especificado
    top_specs = df_year['specs'].explode().value_counts().head(5).index.tolist()
    return top_specs



# Función para obtener la cantidad de juegos lanzados en un año con early access
@app.get('/earlyacces/')
def earlyacces(Año: str):
    año_str = str(Año)
    # Filtrar el DataFrame para el año especificado
    mask = (df['year'].astype(str).str.startswith(año_str)) & (df["early_access"] == True)
    df_year = df[mask]
    df_year = df[mask]

    games = len(df_year)
    return {"games": games}


@app.get('/sentiment/')
def sentiment(Año: int):
    año_str = str(Año)

    # Filtrar el DataFrame para el año especificado
    df_year = df[df['year'].astype(str).str.startswith(año_str)]
    
    # Obtener el análisis de sentimiento y contar la cantidad de registros en cada categoría
    analisis_sentimiento = df_year['sentiment'].value_counts().to_dict()
    
    return analisis_sentimiento

@app.get('/metascore/')
def metascore(Año: int):

    año_str = str(Año)

    # Filtrar el DataFrame para el año especificado
    df_year = df[df['year'].astype(str).str.startswith(año_str)]
    
    # Obtener los top 5 juegos con mayor metascore en el año especificado
    top_metascore_juegos = df_year.nlargest(5, 'metascore')['app_name'].tolist()
    return top_metascore_juegos




# Load the trained model
loaded_model = joblib.load('gradient_boosting.pkl', protocol=4)

#Prediccion de precios, basado en el modelo entrenado y guardado


class GameFeatures(BaseModel):
    metascore: float
    early_acces: bool
    year: int






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



@app.get("/predict")
def predict(metascore: float = None, year: int = None, genre: Genre = None):
     # Validate that all required parameters are provided
    if metascore is None or year is None or genre is None:
        raise HTTPException(status_code=400, detail="Missing parameters")

    # Convert the input to a DataFrame with the columns required for the model
    input_df = pd.DataFrame(
        [[metascore, year, genre == Genre.Early_Access, genre.value == Genre.Free_to_Play]],
        columns=['early_access', 'metascore', 'year', 'genres_encoded']
    )

    # Perform the prediction with the model
    try:
      price = loaded_model.predict(input_df)[0]
    except ValueError as e:
         raise HTTPException(status_code=400, detail="Invalid input: " + str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

    # Return the price and RMSE as output
    return {"price": price, "RMSE del modelo": 8.827512792544086}










