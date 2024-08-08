import streamlit as st
import pandas as pd
import pickle
from preprocessing import main as preprocess_data

#######################
# Load data
X_train_trans, X_test_trans, y_train, y_test, feature_names, X_test_trans_df, testset = preprocess_data()

# Load model
with open('XGB_F191.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to display top features as horizontal bars
def display_top_features_bars(model, feature_names):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(5)  # Top 5 features

    # Remove "MinMax_Q__" prefix from feature names
    feature_importance_df['Feature'] = feature_importance_df['Feature'].str.replace('MinMax_Q__', '', regex=False)

    st.markdown("#### Top 5 Important Features")

    # Create a horizontal bar chart
    max_importance = feature_importance_df['Importance'].max()
    for index, row in feature_importance_df.iterrows():
        st.markdown(f"<div style='font-size: 12px;'><b>{row['Feature']}</b></div>", unsafe_allow_html=True)
        st.progress(int(row['Importance'] / max_importance * 100))

#######################
# Sidebar
with st.sidebar:
    st.title('🔍 Client Prediction Dashboard')

    client_list = testset['CLIENTNUM'].unique()
    selected_client = st.selectbox('Select a client by number', client_list)  

#######################
# Function to display client details and prediction
def display_client_details_and_prediction(client_num):
    if client_num in testset['CLIENTNUM'].values:
        sample_idx = testset.index[testset['CLIENTNUM'] == client_num][0]
        sample_data_original = testset.iloc[sample_idx]

        categorical_vars = ["Gender", "Education_Level", "Marital_Status", "Income_Category"]
        numeric_vars = ["Customer_Age", "Total_Trans_Ct", "Total_Trans_Amt",
                        "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Total_Ct_Chng_Q4_Q1", "Total_Revolving_Bal",
                        "Total_Amt_Chng_Q4_Q1", "Total_Relationship_Count"]

        st.markdown('#### Client Details')
        # Display categorical variables
        cat_cols = st.columns(4)
        for i, var in enumerate(categorical_vars):
            with cat_cols[i % 4]:
                st.markdown(f"**{var.replace('_', ' ')}**:")
                st.info(f"{sample_data_original[var]}")

        num_cols = st.columns(5)
        for i, var in enumerate(numeric_vars):
            min_value = testset[var].min()
            max_value = testset[var].max()
            current_value = sample_data_original[var]
            
            with num_cols[i % 5]:
                st.slider(
                    label=var.replace('_', ' '),
                    min_value=float(min_value),
                    max_value=float(max_value),
                    value=float(current_value),
                    format="%.2f"  # Formatting to two decimal places
                )

        # Predict automatically
        sample_data = X_test_trans[sample_idx].reshape(1, -1)
        prediction = model.predict(sample_data)
        prediction_proba = model.predict_proba(sample_data)[0]  # Get the probabilities for each class

        # Determine result and color
        result = 'Restera' if prediction[0] == 0 else 'Va partir'
        color = 'green' if prediction[0] == 0 else 'red'
        prob = prediction_proba[0] if prediction[0] == 0 else prediction_proba[1]  # Probability for the predicted class

        # Display prediction result with a small box
        st.markdown(f"""
        <div style='
            background-color: {color}; 
            color: white; 
            padding: 10px; 
            border-radius: 5px; 
            width: 150px; 
            text-align: center; 
            font-size: 16px;'>
            <strong>Prediction</strong><br>
            {result}
        </div>
        """, unsafe_allow_html=True)

        # Display probability of confidence with st.metric
        st.metric(label="Probability of Confidence", value=f"{prob:.2%}")

    else:
        st.write("Client not found in the test set.")

#######################
# Dashboard Main Panel
col1, sep, col2 = st.columns((3.5, 0.1, 1), gap='medium')

with col1:
    display_client_details_and_prediction(selected_client)

with col2:
    st.markdown('#### Model performance')
    st.metric(label="F1_Score", value="91%")
    # Display top features as horizontal bars
    display_top_features_bars(model, feature_names)

with sep:
    # Display a vertical line
    st.markdown("<div style='background-color: gray; width: 1px; height: 100vh;'></div>", unsafe_allow_html=True)


    
