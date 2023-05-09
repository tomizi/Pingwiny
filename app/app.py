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

penguins = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv')

st.set_page_config(page_title='App', page_icon = "🧊", layout = 'wide')
st.sidebar.header('Odfiltruj wybrane cechy:')
species = st.sidebar.multiselect(
    'Gatunek',
    options = penguins['species'].unique(),
    default = penguins['species'].unique()
    
    )
island = st.sidebar.multiselect(
    'Wyspa:',
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
    value = int(np.ceil(np.max(penguins['bill_length_mm']))),
    step = 1
    
    )

bill_depth_mm = st.sidebar.slider(
    'Głębokosć dzioba w milimetrach',
    min_value = int(np.ceil(np.min(penguins['bill_depth_mm']))),
    max_value = int(np.ceil(np.max(penguins['bill_depth_mm']))),
    value = int(np.ceil(np.max(penguins['bill_depth_mm']))),
    step = 1
    
    )

flipper_length_mm = st.sidebar.slider(
    'Długosć skrzydła w milimetrach',
    min_value = int(np.ceil(np.min(penguins['flipper_length_mm']))),
    max_value = int(np.ceil(np.max(penguins['flipper_length_mm']))),
    value = int(np.ceil(np.max(penguins['flipper_length_mm']))),
    step = 10
    
    )
#st.header('Witam!')
#st.subheader('Ta strona jest przeznaczona dla wąskiego grona odbirców.')
#L = np.random.choice([1,2,3,4,5,6], size = 100, replace=True, p =[1/6]*6)
#st.markdown('Realizacja procesu stochastycznego **:blue[$X_1,X_2,\ldots,X_{100}$]**, gdzie *$X_i$* ma rozkład opisujący eksperyment rzutu symetrzyczną koscią do gry.')
#st.write(str(list(L)))

st.title('Analiza danych')
st.markdown('---')
st.header('W tabeli poniżej znajdują się dane na temat trzech gatunków pingwina.')
penguins_new = penguins.query('species == @species & island == @island & sex == @sex & body_mass_g >= @masa_ciała[0] & body_mass_g <= @masa_ciała[1] & flipper_length_mm <= @flipper_length_mm & bill_depth_mm <= @bill_depth_mm & bill_length_mm <= @bill_length_mm')
st.write('[Link do zbiorów danych >](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv)')
if penguins_new.empty:
    st.info('Brak danych do wyswietlenia.')
else:
    st.dataframe(penguins_new.style.format({'body_mass_g':"{:.2f}",'bill_length_mm':"{:.2f}",'bill_depth_mm':"{:.2f}",'flipper_length_mm':"{:.2f}"}),use_container_width = True)
    st.dataframe(penguins_new.describe(),use_container_width = True)
    


try:
    fig1 = px.scatter(penguins_new,x='body_mass_g', y='flipper_length_mm',color='sex',symbol='species',trendline='ols',trendline_scope="overall",trendline_color_override="grey",
                    color_discrete_map={'FEMALE':'red','MALE':'blue'},symbol_sequence= ['circle', 'triangle-up', 'square']
                    ).update_xaxes(title='masa ciała [g]'
                    ).update_yaxes(title = 'długosc płetwy [mm]'
                    ).update_layout(title='Wykres rozrzutu',title_x=0.5,title_font_size=25)
    st.plotly_chart(fig1,use_container_width = True)
    results1 = px.get_trendline_results(fig1)
    st.write(results1.px_fit_results.iloc[0].summary())
    #px.scatter(penguins,x='bill_length_mm', y='flipper_length_mm', title='Wykres rozrzutu')
    #px.scatter(penguins_new,x='bill_length_mm', y='flipper_length_mm',color='sex',symbol='species', title='Wykres rozrzutu').update_layout(show_legend=True)
    #px.scatter(penguins,x='bill_length_mm', y='flipper_length_mm',color='sex', title='Wykres rozrzutu',color_discrete_map={'FEMALE':'red','MALE':'blue'})
    fig = px.scatter(penguins_new,x='body_mass_g', y='flipper_length_mm',color='sex',facet_col='island',facet_row='species',trendline='ols',color_discrete_map={'FEMALE':'red','MALE':'blue'}
                    ).update_xaxes(title='masa ciała [g]'
                    ).update_yaxes(title = 'długosc płetwy [mm]'
                    ).update_layout(title='Wykres rozrzutu',title_x=0.5,title_font_size=25,width=1000,height=800)
    st.plotly_chart(fig,use_container_width = True)
    results = px.get_trendline_results(fig)
    #st.write(results.query("island == 'Biscoe' & species == 'Gentoo'").px_fit_results.iloc[0].summary())
    
    
    X = penguins_new.iloc[:,4:5].values
    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 500, n_init = 10, random_state = 123)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
    fig2 = go.Figure(data = go.Scatter(x = [1,2,3,4,5,6,7,8,9,10], y = wcss))
    
    
    fig2.update_layout(title='WCSS vs. Cluster number',
                       xaxis_title='Clusters',
                       yaxis_title='WCSS')
    st.plotly_chart(fig2,use_container_width = True)
    
    
    kmeans = KMeans(n_clusters = 2, init="k-means++", max_iter = 500, n_init = 10, random_state = 123)
    identified_clusters = kmeans.fit_predict(X)
    
    
    data_with_clusters = penguins_new.copy()
    data_with_clusters['Cluster'] = np.where(identified_clusters==1,'1','0')
    
    st.dataframe(data_with_clusters)
    fig3 = px.scatter(data_with_clusters,x='body_mass_g', y='flipper_length_mm',color='Cluster',color_discrete_map={'1':'red','0':'blue'}
                    ).update_xaxes(title='masa ciała [g]'
                    ).update_yaxes(title = 'długosc płetwy [mm]'
                    ).update_layout(title='Wykres rozrzutu',title_x=0.5,title_font_size=25)
    st.plotly_chart(fig3,use_container_width = True)
except:
    st.warning('Nie można przeprowadzić analizy, ponieważ wykryto błąd.')
