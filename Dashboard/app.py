import streamlit as st
import pandas as pd
import pickle
from preprocessing import preprocess_data
from visu import display_top_features_chart, plot_distribution, display_probability_gauge

#######################
### Page Configuration
st.set_page_config(
    page_title="Dashboard Prédiction Client",
    page_icon="🔍",
    layout="wide"
)

#######################
### Chargement des données et du modèle
@st.cache_data
def load_resources():
    # Le preprocessing est mis en cache pour la performance
    X_train_trans, X_test_trans, y_train, y_test, feature_names, X_test_trans_df, testset = preprocess_data()
    
    # Chargement du modèle
    with open('XGB_F191.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Calcul des probabilités
    X_test_trans_df_for_proba = pd.DataFrame(X_test_trans, columns=feature_names)
    testset['Prob_Leave'] = model.predict_proba(X_test_trans_df_for_proba)[:, 1]
    
    # Création de l'indicateur visuel
    testset['Indicator'] = testset['Prob_Leave'].apply(lambda x: '🔴' if x > 0.5 else '🟢')
    
    return X_test_trans, feature_names, testset, model

X_test_trans, feature_names, testset, model = load_resources()

#######################
# Barre latérale (Sidebar)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830289.png", width=100)
    st.title("Dashboard de Prédiction")

    # Sélection du client
    client_list = testset[['CLIENTNUM', 'Indicator']].drop_duplicates()
    client_list['Display'] = client_list.apply(lambda row: f"{row['CLIENTNUM']} {row['Indicator']}", axis=1)
    selected_client_display = st.selectbox('Sélectionnez un client', client_list['Display'])
    
    # Extraction du numéro de client
    selected_client_num = int(selected_client_display.split()[0])

    st.divider()
    
    # Sélection de la variable pour le graphique de distribution
    st.header("Analyse de Distribution")
    exclude_columns = [
        'CLIENTNUM', 'Attrition_Flag', 'Prob_Leave', 'Indicator',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1'
    ]
    variable_list = [col for col in testset.columns if col not in exclude_columns]
    selected_variable = st.selectbox('Sélectionnez une variable', variable_list)

# Récupération des données du client sélectionné
client_data_original = testset[testset['CLIENTNUM'] == selected_client_num].iloc[0]
sample_idx = testset.index[testset['CLIENTNUM'] == selected_client_num][0]
sample_data_transformed = X_test_trans[sample_idx].reshape(1, -1)

#######################
# Interface Principale
st.title(f"🔍 Analyse du Client #{selected_client_num}")
st.markdown("Ce tableau de bord permet d'analyser les détails d'un client, de prédire son risque de départ et de comprendre les facteurs clés de cette prédiction.")

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Prédiction")
    
    prediction = model.predict(sample_data_transformed)
    prediction_proba = model.predict_proba(sample_data_transformed)[0]
    
    if prediction == 1:
        st.error("Risque de départ élevé", icon="🔥")
    else:
        st.success("Client fidèle", icon="✅")

    display_probability_gauge(prediction_proba[1])

with col2:
    st.header("Facteurs d'Influence Clés")
    display_top_features_chart(model, feature_names)

st.divider()

st.header("Profil du Client")

with st.container():
    # Variables catégorielles
    cat_vars = ["Gender", "Education_Level", "Marital_Status", "Income_Category"]
    cat_cols = st.columns(len(cat_vars))
    for i, var in enumerate(cat_vars):
        cat_cols[i].metric(label=var.replace('_', ' '), value=client_data_original[var])
        
    st.text("") 

    # Variables numériques
    numeric_vars = [
        "Customer_Age", "Total_Trans_Ct", "Total_Trans_Amt", "Months_Inactive_12_mon", 
        "Contacts_Count_12_mon", "Total_Ct_Chng_Q4_Q1", "Total_Revolving_Bal", 
        "Total_Amt_Chng_Q4_Q1", "Total_Relationship_Count"
    ]
    
    num_cols_1 = st.columns(5)
    for i, var in enumerate(numeric_vars[:5]):
        num_cols_1[i].metric(label=var.replace('_', ' '), value=f"{client_data_original[var]:.2f}")

    num_cols_2 = st.columns(4)
    for i, var in enumerate(numeric_vars[5:]):
        num_cols_2[i].metric(label=var.replace('_', ' '), value=f"{client_data_original[var]:.2f}")

st.divider()

with st.expander("Voir l'analyse de distribution comparative"):
    plot_distribution(testset, selected_variable, client_data_original)