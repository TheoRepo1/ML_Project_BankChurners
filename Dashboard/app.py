import streamlit as st
import pandas as pd
import pickle
from preprocessing import preprocess_data
from visu import display_top_features_bars, plot_distribution

# Load data
X_train_trans, X_test_trans, y_train, y_test, feature_names, X_test_trans_df, testset = preprocess_data()

# Load model
with open('XGB_F191.pkl', 'rb') as file:
    model = pickle.load(file)

# Calculate probabilities
X_test_trans_df = pd.DataFrame(X_test_trans, columns=feature_names)
testset['Prob_Leave'] = model.predict_proba(X_test_trans_df)[:, 1]

# Create a column for visual indicator
testset['Indicator'] = testset['Prob_Leave'].apply(lambda x: '🔴' if x > 0.5 else '')

#######################
# Sidebar
with st.sidebar:
    st.title('Dashboard by Théo JEAN')

    # Select client by number with indicator
    client_list = testset[['CLIENTNUM', 'Indicator']].drop_duplicates()
    client_list['Display'] = client_list.apply(lambda row: f"{row['CLIENTNUM']} {row['Indicator']}", axis=1)
    selected_client = st.selectbox('Select a client by number', client_list['Display'])

    # Extract the client number from the selected option
    selected_client_num = selected_client.split()[0]

    # Select variable for distribution plot
    exclude_columns = ['CLIENTNUM', 'Attrition_Flag', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
                       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1']
    variable_list = [col for col in testset.columns if col not in exclude_columns]
    selected_variable = st.selectbox('Select a variable for distribution plot', variable_list)

    if selected_client and selected_variable:
        client_data = testset[testset['CLIENTNUM'] == int(selected_client_num)].iloc[0]
        plot_distribution(testset, selected_variable, client_data)

#######################
# Main Dashboard Layout
col0, col1 = st.columns([2, 1], gap='large')

with col0:
    st.header("🔍 Client Prediction Dashboard")
    
    # Display Client Details
    if selected_client_num:
        sample_idx = testset.index[testset['CLIENTNUM'] == int(selected_client_num)][0]
        sample_data_original = testset.iloc[sample_idx]

        categorical_vars = ["Gender", "Education_Level", "Marital_Status", "Income_Category"]
        numeric_vars = ["Customer_Age", "Total_Trans_Ct", "Total_Trans_Amt",
                        "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Total_Ct_Chng_Q4_Q1", "Total_Revolving_Bal",
                        "Total_Amt_Chng_Q4_Q1", "Total_Relationship_Count"]

        st.markdown('#### Client Details')
        
        # Display categorical variables in a row
        cat_cols = st.columns(4)
        for i, var in enumerate(categorical_vars):
            with cat_cols[i % 4]:
                st.markdown(f"<div style='width: 150px;'><strong>{var.replace('_', ' ')}:</strong></div>", unsafe_allow_html=True)
                st.info(f"{sample_data_original[var]}")

        # Display numeric variables in rows of 4
        for i in range(0, len(numeric_vars), 4):
            num_cols = st.columns(4)
            for j, var in enumerate(numeric_vars[i:i+4]):
                min_value = testset[var].min()
                max_value = testset[var].max()
                current_value = sample_data_original[var]

                with num_cols[j]:
                    st.slider(
                        label=f"**{var.replace('_', ' ')}**",
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=float(current_value),
                        format="%.2f"
                    )

with col1:
    # Display Prediction and Probabilities
    if selected_client_num:
        sample_idx = testset.index[testset['CLIENTNUM'] == int(selected_client_num)][0]
        sample_data = X_test_trans[sample_idx].reshape(1, -1)
        prediction = model.predict(sample_data)
        prediction_proba = model.predict_proba(sample_data)[0]

        # Display prediction result
        st.markdown('#### Prediction Probability')
        if prediction == 1:
            st.error(f"The model predicts that this client will **leave** the bank.")
        else:
            st.success(f"The model predicts that this client will **stay** with the bank.")

        # Display prediction probabilities
        st.info(f"Probability of leaving: **{prediction_proba[1]:.2%}**")

        # Display top 5 important features
        display_top_features_bars(model, feature_names)
