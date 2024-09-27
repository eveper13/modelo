import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import nltk
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Descargar los recursos necesarios para VADER
nltk.download('vader_lexicon')

# Clase para el análisis de sentimientos
class SentimentAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.apply(self.get_sentiment_label)
    
    def get_sentiment_label(self, text):
        sentiment_score = self.sia.polarity_scores(text)['compound']
        if sentiment_score >= 0.05:
            return 'positive'
        elif sentiment_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

# Función para cargar los datos desde un CSV
@st.cache_data
def load_data():
    with st.spinner('Cargando datos...'):
        return pd.read_csv('negative_reviews.csv')

# Función para filtrar reseñas negativas
@st.cache_data  # Cambiado de st.cache a st.cache_data
def filter_negative_reviews(data):
    return data[data['sentiment_label'] == 'negative']

# Función para detectar patrones de insatisfacción
def detect_dissatisfaction_patterns(data):
    # Definir palabras clave relacionadas con las quejas en inglés
    keywords = {
        'Falta de opciones de comida': ['few options', 'no options', 'lack of variety', 'limited options', 'not enough choices'],
        'Servicio lento': ['slow', 'took a long time', 'delayed', 'long wait'],
        'Falta de personal': ['few staff', 'lack of employees', 'not enough staff'],
        'Problemas de entrega': ['poor delivery service', 'late delivery', 'delivery problems']
    }
    
    dissatisfaction_counts = {key: 0 for key in keywords.keys()}
    
    for key, phrases in keywords.items():
        for phrase in phrases:
            dissatisfaction_counts[key] += data['text'].str.contains(phrase, case=False, na=False).sum()
    
    return dissatisfaction_counts

# Función para mostrar reseñas negativas por ciudad
def show_negative_reviews_by_city(city, data):
    city_negative_reviews = data[data['city'] == city]
    
    if city_negative_reviews.empty:
        st.write(f"No hay reseñas negativas para la ciudad: {city}")
        return
    
    # Contar reseñas negativas por restaurante
    negative_review_counts = city_negative_reviews['name'].value_counts().reset_index()
    negative_review_counts.columns = ['restaurant', 'negative_review_count']
    
    # Identificar el restaurante con más reseñas negativas
    top_restaurant = negative_review_counts.iloc[0]['restaurant']
    
    # Filtrar las reseñas negativas para ese restaurante
    top_restaurant_reviews = city_negative_reviews[city_negative_reviews['name'] == top_restaurant]
    
    # Seleccionar hasta 5 reseñas negativas del restaurante con más reseñas
    limited_reviews = top_restaurant_reviews.head(5)

    # Graficar la cantidad de reseñas negativas para los restaurantes
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='negative_review_count', y='restaurant', data=negative_review_counts.head(10), ax=ax)
    ax.set_title(f'Cantidad de Reseñas Negativas por Restaurante en {city}')
    ax.set_xlabel('Cantidad de Reseñas Negativas')
    ax.set_ylabel('Restaurante')
    st.pyplot(fig)

    # Mostrar ejemplos de reseñas negativas
    st.write(f"Reseñas Negativas para {top_restaurant} en {city}")
    st.write(limited_reviews[['name', 'text']])

    # Mostrar patrones de insatisfacción
    dissatisfaction_counts = detect_dissatisfaction_patterns(city_negative_reviews)
    st.write("Patrones de insatisfacción detectados:")
    
    # Graficar patrones de insatisfacción
    dissatisfaction_df = pd.DataFrame(list(dissatisfaction_counts.items()), columns=['Patrón', 'Cantidad'])
    
    # Graficar patrones de insatisfacción
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Cantidad', y='Patrón', data=dissatisfaction_df, palette='viridis', ax=ax)
    ax.set_title('Patrones de Insatisfacción Detectados')
    ax.set_xlabel('Cantidad de Quejas')
    ax.set_ylabel('Patrón')
    st.pyplot(fig)

# Interfaz de usuario principal
def main():
    st.title('Análisis de Reseñas')
    
    # Cargar los datos
    yelp_reviews = load_data()
    
    # Filtrar reseñas negativas
    negative_reviews = filter_negative_reviews(yelp_reviews)

    if 'city' not in negative_reviews.columns:
        st.write("Error: La columna 'city' no se encuentra en los datos.")
        return

    # Gráfico de barras para las 10 ciudades con más reseñas negativas
    city_counts = negative_reviews['city'].value_counts().reset_index()
    city_counts.columns = ['city', 'negative_review_count']
    
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.barplot(x='negative_review_count', y='city', data=city_counts.head(10), hue='city', palette='coolwarm', legend=False)
    plt.title('Top 10 Ciudades con Más Reseñas Negativas', fontsize=16, fontweight='bold')
    plt.xlabel('Cantidad de Reseñas Negativas', fontsize=12)
    plt.ylabel('Ciudad', fontsize=12)

    # Añadir anotaciones para cada barra (valores exactos)
    for index, value in enumerate(city_counts['negative_review_count'].head(10)):
        plt.text(value + 0.5, index, str(value), color='black', ha="center", va="center", fontweight='bold')

    plt.tight_layout()
    st.pyplot(plt)

    city_input = st.text_input("Introduce el nombre de la ciudad (ej. 'Reno'):")
    
    if st.button("Mostrar Reseñas Negativas"):
        if city_input:
            show_negative_reviews_by_city(city_input, negative_reviews)
        else:
            st.warning("Por favor, introduce una ciudad.")

if __name__ == "__main__":
    main()
