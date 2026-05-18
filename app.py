import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predicción de géneros musicales", layout="wide")

modelo = joblib.load("mejor_modelo_genre.pkl")
le = joblib.load("label_encoder_genre.pkl")

st.title("Aplicación de predicción de género musical")
st.write("Esta app predice el género musical usando características de canciones.")

acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
key = st.slider("Key", 0, 11, 5)
liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
loudness = st.slider("Loudness", -60.0, 5.0, -10.0)
mode = st.slider("Mode", 0, 1, 1)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
tempo = st.slider("Tempo", 0.0, 250.0, 120.0)
timesignature = st.slider("Time Signature", 0, 7, 4)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
year = st.slider("Year", 1986, 2023, 2010)
durationmin = st.slider("Duration (min)", 0.0, 15.0, 3.5)

entrada = pd.DataFrame([{
    "acousticness": acousticness,
    "danceability": danceability,
    "energy": energy,
    "instrumentalness": instrumentalness,
    "key": key,
    "liveness": liveness,
    "loudness": loudness,
    "mode": mode,
    "speechiness": speechiness,
    "tempo": tempo,
    "time_signature": timesignature,
    "valence": valence,
    "year": year,
    "duratio_nmin": durationmin
}])

if st.button("Predecir género"):
    pred = modelo.predict(entrada)
    genero = le.inverse_transform(pred)
    st.success(f"Género predicho: {genero[0]}")
