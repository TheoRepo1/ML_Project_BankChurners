import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
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

    # Select variable for distribution plot
    variable_list = testset.columns.tolist()
    selected_variable = st.selectbox('Select a variable for distribution plot', variable_list)

    # Function to plot distribution of selected variable
    def plot_distribution(variable, client_data):
        value_counts = testset[variable].value_counts().reset_index()
        value_counts.columns = [variable, 'Count']

        fig = px.bar(value_counts, x=variable, y='Count', title=f'Distribution of {variable}', labels={variable: variable, 'Count': 'Count'})

        # Highlight the client's value with a small red bar
        client_value = client_data[variable]
        client_count = value_counts[value_counts[variable] == client_value]['Count'].values[0]
        fig.add_trace(go.Bar(x=[client_value], y=[client_count], marker=dict(color='red'), name='Client Value', width=0.2))

        # Adjust y-axis to fit the data
        fig.update_layout(yaxis=dict(range=[0, value_counts['Count'].max() * 1.1]))

        st.plotly_chart(fig)

    # Plot distribution of selected variable
    if selected_client and selected_variable:
        client_data = testset[testset['CLIENTNUM'] == selected_client].iloc[0]
        plot_distribution(selected_variable, client_data)

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
                st.markdown(f"<div style='width: 150px;'><strong>{var.replace('_', ' ')}:</strong></div>", unsafe_allow_html=True)
                st.info(f"{sample_data_original[var]}")

        # Display numeric variables in groups of 4
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

        return result, color, prob

    else:
        st.write("Client not found in the test set.")
        return None, None, None

#######################
# Dashboard Main Panel
col0, sep, col1 = st.columns((3, 0.1, 1), gap='medium')

with col0:
    result, color, prob = display_client_details_and_prediction(selected_client)

with col1:
    if result:
        st.markdown('#### Prediction')
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
        st.metric(label="Confidence", value=f"{prob:.2%}")
        display_top_features_bars(model, feature_names)

with sep:
    # Display a vertical line
    st.markdown("<div style='background-color: gray; width: 1px; height: 100vh;'></div>", unsafe_allow_html=True)
