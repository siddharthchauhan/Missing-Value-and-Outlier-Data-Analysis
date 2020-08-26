import streamlit as st
import pandas as pd
import numpy as np
#import io
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.anomaly import load_model, predict_model
import plotly.express as px
import plotly.graph_objects as go

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Missing Value and Outlier Data Analysis")
st.sidebar.title("Missing data Analysis")

st.markdown("This application is a dashboard to analyze the missing and Outlier Data Analysis")

upload_data = st.sidebar.file_uploader("Upload a Dataset", type=["csv","txt"])
#text_io = io.TextIOWrapper(upload_data)

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

    # Display the percentage of missing values

    if st.sidebar.checkbox("Display percentage of missing values"):
        st.markdown("## Percentage(%) of missing values")
        total_missing_value = df.isna().sum()
        shape_rows = len(df)
        percent_value_column = (total_missing_value/shape_rows)*100
        #display_percentage_df = pd.DataFrame(percent_value_column, columns =['Percent'])
        display_percentage_df = percent_value_column.to_frame()
        display_percentage_df.columns = ['Percent']
        display_percentage_df.index.names = ['Name']
        display_percentage_df['Name'] = data_columns
        display_percentage_df.reset_index(drop=True, inplace=True)
        display_percentage_df.sort_values(by=['Percent'],ascending=False,inplace=True)
        cols = ['Name','Percent']
        st.write(display_percentage_df[cols])
        #st.write(display_percentage_df)

        # Pie plot for % of missing values

        if st.sidebar.checkbox("Plot percentage of missing values for all columns"):
            st.markdown("## Pie plot for percentage of missing values")
            #Transpose_df = display_percentage_df.T 
            #st.write(Transpose_df)
            # chart_pie = display_percentage_df.plot(y='Percent', kind='pie', legend=False, use_index=True, figsize=(3, 3),
            # sort_columns=True, ylabel=display_percentage_df.columns)
            chart_pie = px.pie(display_percentage_df, values='Percent', names='Name', labels={'Name':'Percent'})
            chart_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(chart_pie)
        
    
    # Bar plot for number of missing values

    if st.sidebar.checkbox("Plot total missing records from all columns"):
        st.markdown("## Bar plot for number of missing values")
        #chart = missing_df.plot.bar(x='Name', y='Count', rot=90, figsize=(10,6),use_index =True, sort_columns=True, legend=True)
        chart = px.bar(missing_df, x='Name', y='Count', color= 'Count')
        st.plotly_chart(chart)
        #st.write(chart)
        #st.pyplot()

    


st.sidebar.title("Outlier data Analysis")

upload_data_out = st.sidebar.file_uploader("Upload a Dataset to detect outlier", type=["csv","txt"])

if upload_data_out is not None:
    df_out = pd.read_csv(upload_data_out)
    st.dataframe(df_out)

    if df_out["DOMAIN"][0] == 'DM':
        predictions = predict_model(model_1, data=df_out)
            
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df_pred = pd.DataFrame(predictions)

        fig = px.scatter(df_pred, x ='Score',y='Score',color='Label')
    
        st.plotly_chart(fig)

        
    if df_out["DOMAIN"][0] == 'DS':
        predictions = predict_model(model_2, data=df_out)
            
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df_pred = pd.DataFrame(predictions)

        fig = px.scatter(df_pred, x ='Score',y='Score',color='Label')

        st.plotly_chart(fig)


    if df_out["DOMAIN"][0] == 'MI':
        predictions = predict_model(model_3, data=df_out)
            
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df_pred = pd.DataFrame(predictions)

        fig = px.scatter(df_pred, x ='Score',y='Score',color='Label')

        st.plotly_chart(fig)

    if df_out["DOMAIN"][0] == 'PR':
        predictions = predict_model(model_4, data=df_out)
            
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df_pred = pd.DataFrame(predictions)

        fig = px.scatter(df_pred, x ='Score',y='Score',color='Label')

        st.plotly_chart(fig)
        
    if df_out["DOMAIN"][0] == 'SS':
        predictions = predict_model(model_5, data=df_out)
            
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df_pred = pd.DataFrame(predictions)

        fig = px.scatter(df_pred, x ='Score',y='Score',color='Label')

        st.plotly_chart(fig)
        
    if df_out["DOMAIN"][0] == 'TR':
        predictions = predict_model(model_6, data=df_out)
            
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df_pred = pd.DataFrame(predictions)

        fig = px.scatter(df_pred, x ='Score',y='Score',color='Label')

        st.plotly_chart(fig)

    if df_out["DOMAIN"][0] == 'TU':
        predictions = predict_model(model_7, data=df_out)
            
        st.markdown("## Potential Outliers")

        # The outlier's in a dataset are
        outlier = predictions['Label'] == 1
        st.write(predictions[outlier])

        df_pred = pd.DataFrame(predictions)

        fig = px.scatter(df_pred, x ='Score',y='Score',color='Label')

        st.plotly_chart(fig)
