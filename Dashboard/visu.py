import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def display_top_features_chart(model, feature_names):
    """
    Affiche les 5 features les plus importantes via un graphique à barres Plotly.
    """
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True).tail(5)

    feature_importance_df['Feature'] = feature_importance_df['Feature'].str.replace('MinMax_Q__', '', regex=False).str.replace('_', ' ')

    fig = go.Figure(go.Bar(
        x=feature_importance_df['Importance'],
        y=feature_importance_df['Feature'],
        orientation='h',
        marker_color='#00B0F0'
    ))
    
    fig.update_layout(
        title="Top 5 des Facteurs de Prédiction",
        xaxis_title="Importance",
        yaxis_title="Facteur",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

def display_probability_gauge(probability):
    """
    MODIFIÉ: Affiche la probabilité sur une échelle de 0 à 1.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,  # MODIFIÉ: On utilise la valeur directe (0-1)
        number={'valueformat': '.2%'}, # Affiche la proba en pourcentage (ex: 0.58 -> 58.00%)
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de Départ", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"}, # MODIFIÉ: Échelle de 0 à 1
            'bar': {'color': "#FF4B4B" if probability > 0.5 else "#2ECC71"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.25], 'color': 'lightgreen'},   # MODIFIÉ
                {'range': [0.25, 0.5], 'color': 'lightyellow'} # MODIFIÉ
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5  # MODIFIÉ
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

def plot_distribution(df, variable, client_data):
    """
    Gère les variables numériques et catégorielles.
    - Pour les nombres : dessine une ligne verticale.
    - Pour le texte : colore la barre correspondante en rouge.
    """
    client_value = client_data[variable]

    # Détecte si la variable est numérique
    if pd.api.types.is_numeric_dtype(df[variable]):
        # Comportement original pour les données numériques
        fig = px.histogram(df, x=variable, title=f'Distribution of {variable}', nbins=30)
        fig.add_vline(x=client_value, line_width=3, line_dash="dash", line_color="red",
                      annotation_text="Client", annotation_position="top right")
    else:
        # Nouveau comportement pour les données catégorielles (texte)
        value_counts = df[variable].value_counts().sort_index()
        
        # Crée une liste de couleurs pour mettre en évidence la barre du client
        colors = ['#FF4B4B' if cat == client_value else '#00B0F0' for cat in value_counts.index]

        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f'Distribution of {variable}',
            labels={'x': variable, 'y': 'Count'},
            color=colors,
            color_discrete_map="identity" # Indique à Plotly d'utiliser les couleurs telles quelles
        )
    
    fig.update_layout(showlegend=False) # Masque la légende pour plus de clarté
    st.plotly_chart(fig, use_container_width=True)