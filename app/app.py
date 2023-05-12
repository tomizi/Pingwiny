# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 19:10:40 2023

@author: User
"""
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from PIL import Image
import requests

im = Image.open(requests.get("https://static.wikia.nocookie.net/pingwiny-z-madagaskaru-fanfakty/images/3/39/Pingwiny_szeregowy.jpg/revision/latest?cb=20210718232107&path-prefix=pl",stream=True).raw)
#im = Image.open("C:\\Users\\User\\OneDrive - Uniwersytet Mikołaja Kopernika w Toruniu\\Pulpit\\szeregowy.webp")
penguins = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv')

st.set_page_config(page_title='Pingwiny', page_icon = im, layout = 'wide')
st.sidebar.header('Odfiltruj wybrane cechy:')

species = st.sidebar.multiselect(
    'Gatunek',
    options = penguins['species'].unique(),
    default = penguins['species'].unique()
    
    )
island = st.sidebar.multiselect(
    'Wyspa',
    options = penguins['island'].unique(),
    default = penguins['island'].unique()
    
    )
sex = st.sidebar.multiselect(
    'Płeć',
    options = penguins['sex'].unique(),
    default = penguins['sex'].unique()
    
    )

masa_ciała = st.sidebar.slider(
    'Masa ciała w gramach',
    min_value = int(np.ceil(np.min(penguins['body_mass_g']))),
    max_value = int(np.ceil(np.max(penguins['body_mass_g']))),
    value = [int(np.ceil(np.min(penguins['body_mass_g']))), int(np.ceil(np.max(penguins['body_mass_g'])))],
    step = 10
    
    )

bill_length_mm = st.sidebar.slider(
    'Długosć dzioba w milimetrach',
    min_value = int(np.ceil(np.min(penguins['bill_length_mm']))),
    max_value = int(np.ceil(np.max(penguins['bill_length_mm']))),
    value = [int(np.ceil(np.min(penguins['bill_length_mm']))),int(np.ceil(np.max(penguins['bill_length_mm'])))],
    step = 1
    
    )

bill_depth_mm = st.sidebar.slider(
    'Głębokosć dzioba w milimetrach',
    min_value = int(np.ceil(np.min(penguins['bill_depth_mm']))),
    max_value = int(np.ceil(np.max(penguins['bill_depth_mm']))),
    value = [int(np.ceil(np.min(penguins['bill_depth_mm']))),int(np.ceil(np.max(penguins['bill_depth_mm'])))],
    step = 1
    
    )

flipper_length_mm = st.sidebar.slider(
    'Długosć skrzydła w milimetrach',
    min_value = int(np.ceil(np.min(penguins['flipper_length_mm']))),
    max_value = int(np.ceil(np.max(penguins['flipper_length_mm']))),
    value = [int(np.ceil(np.min(penguins['flipper_length_mm']))),int(np.ceil(np.max(penguins['flipper_length_mm'])))],
    step = 1
    
    )

st.title('Analiza danych')
st.markdown('---')
st.subheader('Tabela z danymi o trzech gatunkach pingwina')
penguins_new = penguins.query('species == @species & island == @island & sex == @sex & body_mass_g >= @masa_ciała[0] & body_mass_g <= @masa_ciała[1] & flipper_length_mm >= @flipper_length_mm[0] & flipper_length_mm <= @flipper_length_mm[1] & bill_depth_mm >= @bill_depth_mm[0] & bill_depth_mm <= @bill_depth_mm[1] & bill_length_mm >= @bill_length_mm[0] & bill_length_mm <= @bill_length_mm[1]')
st.write('[Link do zbiorów danych >](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv)')
if penguins_new.empty:
    st.info('Brak danych do wyswietlenia')
else:
    st.dataframe(penguins_new.style.format({'body_mass_g':"{:.2f}",'bill_length_mm':"{:.2f}",'bill_depth_mm':"{:.2f}",'flipper_length_mm':"{:.2f}"}),use_container_width = True)
    st.subheader('Tabela z podstawowymi statystykami')
    st.dataframe(penguins_new.describe(),use_container_width = True)
    


try:
    st.markdown('---')
    st.header('Regresja liniowa')
    st.markdown('---')
    st.subheader('Dopasowanie prostej do danych')
    fig1 = px.scatter(penguins_new,x='body_mass_g', y='flipper_length_mm',color='sex',symbol='species',trendline='ols',trendline_scope="overall",trendline_color_override="grey",
                    color_discrete_map={'FEMALE':'red','MALE':'blue'},symbol_sequence= ['circle', 'triangle-up', 'square']
                    ).update_xaxes(title='masa ciała [g]'
                    ).update_yaxes(title = 'długosc płetwy [mm]'
                    ).update_layout(title='Wykres rozrzutu',title_x=0.5,title_font_size=25)
    st.plotly_chart(fig1,use_container_width = True)
    results1 = px.get_trendline_results(fig1)
    st.subheader('Raport z analizy')
    st.write(results1.px_fit_results.iloc[0].summary())
except:
    st.warning('Nie można przeprowadzić analizy, ponieważ wykryto błąd.')    
try:    
    st.markdown('---')
    st.header('Grupowanie metodą k-średnich')
    st.markdown('---')
    X = penguins_new.iloc[:,4:5].values
    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 500, n_init = 10, random_state = 123)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
    st.subheader('Wykres łokcia')
    fig2 = go.Figure(data = go.Scatter(x = [1,2,3,4,5,6,7,8,9,10], y = wcss))
    fig2.update_layout(title='WCSS vs. Cluster number',
                       xaxis_title='Clusters',
                       yaxis_title='WCSS')
    st.plotly_chart(fig2,use_container_width = True)
    
    n_clu = st.number_input('Wybierz liczbę grup',min_value = 1, max_value = 10,
                            step = 1, value = 2)
    kmeans = KMeans(n_clusters = n_clu, init="k-means++", max_iter = 500, n_init = 10, random_state = 123)
    identified_clusters = kmeans.fit_predict(X)
    
    
    
    data_with_clusters = penguins_new.copy()
    data_with_clusters['Cluster'] = identified_clusters
    for i in range(n_clu):
        data_with_clusters['Cluster'][data_with_clusters['Cluster'] == i] = str(i)

    
    st.subheader('Przewidziane grupy')
    fig3 = px.scatter(data_with_clusters,x='body_mass_g', y='flipper_length_mm',color='Cluster'#,color_discrete_map={'1':'red','0':'blue'}
                    ).update_xaxes(title='masa ciała [g]'
                    ).update_yaxes(title = 'długosc płetwy [mm]'
                    ).update_layout(title='Wykres rozrzutu',title_x=0.5,title_font_size=25)
    st.plotly_chart(fig3,use_container_width = True)
except:
    st.warning('Nie można przeprowadzić analizy, ponieważ wykryto błąd.')
