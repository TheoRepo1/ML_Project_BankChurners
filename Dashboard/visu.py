import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def display_top_features_bars(model, feature_names):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(5)

    feature_importance_df['Feature'] = feature_importance_df['Feature'].str.replace('MinMax_Q__', '', regex=False)

    st.markdown("#### Top 5 Important Features")
    max_importance = feature_importance_df['Importance'].max()
    for index, row in feature_importance_df.iterrows():
        st.markdown(f"<div style='font-size: 12px;'><b>{row['Feature']}</b></div>", unsafe_allow_html=True)
        st.progress(int(row['Importance'] / max_importance * 100))

def plot_distribution(df, variable, client_data, exclude_columns=None):
    """
    Plots the distribution of a selected variable and highlights the client's value.
    """
    if exclude_columns is None:
        exclude_columns = []

    if variable in exclude_columns:
        raise ValueError(f"The selected variable '{variable}' is in the list of excluded columns.")

    value_counts = df[variable].value_counts().reset_index()
    value_counts.columns = [variable, 'Count']

    fig = px.bar(value_counts, x=variable, y='Count', title=f'Distribution of {variable}', labels={variable: variable, 'Count': 'Count'})

    # Highlight the client's value with a small red bar
    client_value = client_data[variable]
    client_count = value_counts[value_counts[variable] == client_value]['Count'].values[0]
    fig.add_trace(go.Bar(x=[client_value], y=[client_count], marker=dict(color='red'), name='Client Value', width=0.2))

    # Adjust y-axis to fit the data
    fig.update_layout(yaxis=dict(range=[0, value_counts['Count'].max() * 1.1]))

    st.plotly_chart(fig)
