import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.anomaly import *
import plotly.express as px
import plotly.graph_objects as go

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Missing Value and Outlier Data Analysis")
st.sidebar.title("Missing and Outlier data Analysis")

st.markdown("This application is a dashboard to analyze the missing and Outlier Data Analysis 🧑‍🔬")

upload_data = st.sidebar.file_uploader("Upload a Dataset", type=["csv","txt"])
text_io = io.TextIOWrapper(upload_data)

model_1 = load_model('DM_SVM_Breast_Cancer')
model_2 = load_model('DS_SVM_Breast_Cancer')
model_3 = load_model('MI_SVM_Breast_Cancer')
model_4 = load_model('PR_SVM_Breast_Cancer')
model_5 = load_model('SS_SVM_Breast_Cancer')
model_6 = load_model('TR_KNN_Breast_Cancer')
model_7 = load_model('TU_KNN_Breast_Cancer')

if upload_data is not None:
    df = pd.read_csv(upload_data)
    st.dataframe(df)

    data_columns= df.columns.tolist()

    # Display count of missing values in dataset

    display_sum_missing_values = df.isna().sum()
    missing_df = display_sum_missing_values.to_frame()
    missing_df.columns = ['Count']
    missing_df.index.names = ['Name']
    missing_df['Name'] = data_columns
    missing_df.reset_index(drop=True, inplace=True)
    missing_df.sort_values(by=['Count'],ascending=False,inplace=True)

    if st.sidebar.checkbox("Display total number of missing values"):
        st.markdown("## Total number of Missing values in whole dataset")
        st.dataframe(missing_df)
  
    # Display the records indexes of missing values

    if st.sidebar.checkbox("Display row index with missing value"):
        st.markdown("##  Missing values indexes from whole dataset")
        display_miss_records_index = missing_df[missing_df.isna()].index.tolist()
        st.dataframe(display_miss_records_index)
 
    # Display the missing value records

    if st.sidebar.checkbox("Display rows with missing values"):
        st.markdown("## Rows with missing values")
        display_missing_records = df[df.isna().any(axis=1)]
        st.dataframe(display_missing_records)
    
    # Bar plot for number of missing values

    if st.sidebar.checkbox("Plot total missing records from all columns"):
        st.markdown("## Bar plot for number of missing values")
        chart = missing_df.plot.bar(x='Name', y='Count', rot=90, figsize=(10,6),use_index =True, sort_columns=True, legend=True)
        st.write(chart)
        st.pyplot()

    
    if df["DOMAIN"][0] == 'DM':
        predictions = predict_model(model_1, data=df)
        
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df = pd.DataFrame(predictions)

        fig = px.scatter(df, x ='Score',y='Score',color='Label')
 
        st.plotly_chart(fig)

       

    elif df["DOMAIN"][0] == 'DS':
        predictions = predict_model(model_2, data=df)
        
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df = pd.DataFrame(predictions)

        fig = px.scatter(df, x ='Score',y='Score',color='Label')

        st.plotly_chart(fig)


    elif df["DOMAIN"][0] == 'MI':
        predictions = predict_model(model_2, data=df)
        
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df = pd.DataFrame(predictions)

        fig = px.scatter(df, x ='Score',y='Score',color='Label')

        st.plotly_chart(fig)

    elif df["DOMAIN"][0] == 'PR':
        predictions = predict_model(model_2, data=df)
        
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df = pd.DataFrame(predictions)

        fig = px.scatter(df, x ='Score',y='Score',color='Label')

        st.plotly_chart(fig)
    
    elif df["DOMAIN"][0] == 'SS':
        predictions = predict_model(model_2, data=df)
        
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df = pd.DataFrame(predictions)

        fig = px.scatter(df, x ='Score',y='Score',color='Label')

        st.plotly_chart(fig)
    
    elif df["DOMAIN"][0] == 'TR':
        predictions = predict_model(model_2, data=df)
        
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df = pd.DataFrame(predictions)

        fig = px.scatter(df, x ='Score',y='Score',color='Label')

        st.plotly_chart(fig)

    elif df["DOMAIN"][0] == 'TU':
        predictions = predict_model(model_2, data=df)
        
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df = pd.DataFrame(predictions)

        fig = px.scatter(df, x ='Score',y='Score',color='Label')

        st.plotly_chart(fig)