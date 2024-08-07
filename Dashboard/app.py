import streamlit as st
import pandas as pd
import numpy as np
import pickle
from preprocessing import main as preprocess_data

# Charger les données prétraitées et non transformées
X_train_trans, X_test_trans, y_train, y_test, feature_names, X_test_trans_df, testset = preprocess_data()

# Charger le modèle
with open('XGB_F191.pkl', 'rb') as file:
    model = pickle.load(file)

# En-tête
st.title("Analyse de l'Attrition Bancaire")
st.write("Ce dashboard présente une analyse approfondie de l'attrition des clients bancaires ainsi qu'une prédiction basée sur un modèle XGBoost.")

# Section Prédiction et Analyse Individuelle
st.header("Prédiction et Analyse Individuelle")
st.write("Sélectionnez un client pour prédire s'il quittera la banque ou non.")

# Sélection d'un client par CLIENTNUM
client_num = st.selectbox("Choisissez un client par numéro", testset['CLIENTNUM'])
cols = ["Customer_Age", "Gender", "Dependent_count", "Education_Level", "Marital_Status",
        "Income_Category", "Card_Category", "Months_on_book", "Total_Relationship_Count",
        "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
        "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1"]

# Vérifiez que l'index existe dans testset
if client_num in testset['CLIENTNUM'].values:
    sample_idx = testset.index[testset['CLIENTNUM'] == client_num][0]
    sample_data = X_test_trans[sample_idx].reshape(1, -1)
    sample_data_original = testset.iloc[sample_idx][cols]

    # Affichage des caractéristiques du client sélectionné
    st.write("Caractéristiques du client sélectionné (non transformées) :")
    st.write(sample_data_original)

    if st.button("Prédire"):
        prediction = model.predict(sample_data)
        st.write("Prédiction :", "Quittera la banque" if prediction[0] == 1 else "Restera à la banque")
else:
    st.write("Client non trouvé dans l'ensemble de test.")
