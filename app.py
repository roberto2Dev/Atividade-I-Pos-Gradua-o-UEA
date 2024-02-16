# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 23:56:01 2024
-----------------------
Use o comando para instalar as bibliotecas essenciais para o projeto:
>>>pip instalar pandas streamlit matplotlib seaborn plotly-express 
>>pip instalar -U scikit-learn #sklearn
-----------------------
Use o comando para iniciar a pagina 
>>>>python -m streamlit run app.py
-----------------------
@author: Roberto Parente
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

import plotly.express as px
import numpy as np

from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from funcoes import carrega_dataset
from PIL import Image #para importar a iamgem logo = Image.open(logo_path)


#-----------------------------------------

# You can always call this function where ever you want
st.set_page_config(layout="wide") # Configurando o layout da pagina 

# adicionando logo no sidebar
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="img/uea-logo.png", width=450, height=260)
st.sidebar.image(my_logo)
# --------------------------


# Configurando o layout do sidebar
st.sidebar.title(':blue[DashBoard]')

st.title('UEA-Universidade do Estado do Amazonas')
col1, col2, col3 = st.columns(3)

with col1:
    
        # adicionando foto do grupo 
    def add_logo(logo_path, width, height):
        """Read and return a resized logo"""
        logo = Image.open(logo_path)
        modified_logo = logo.resize((width, height))
        return modified_logo

    my_logo = add_logo(logo_path="img/grupo.jpeg", width=450, height=260 )
    st.image(my_logo)
with col3:
    st.write(' ')
with col2:

    st.write(' ')
# --------------------------
  #Nome do grupo
    st.markdown("""
                    - Siomar Alencar;
                    - Jameson Souza; 
                    - Odinelma Nascimento;
                    - Marcio Sousa;
                    - Suzana Kamimura;
                    - Roberto Parente;
                    - Marco Falcão;


                """)  

data = carrega_dataset() #Carregando os dados CSV com o arquivo funcao.py

# Tratamento dos dados
# Converter dados para formato numérico
data['Usage_kWh'] = pd.to_numeric(data['Usage_kWh'])
data['Lagging_Current_Reactive.Power_kVarh'] = pd.to_numeric(data['Lagging_Current_Reactive.Power_kVarh'])
data['Leading_Current_Reactive_Power_kVarh'] = pd.to_numeric(data['Leading_Current_Reactive_Power_kVarh'])
data['CO2(tCO2)'] = pd.to_numeric(data['CO2(tCO2)'])
data['Lagging_Current_Power_Factor'] = pd.to_numeric(data['Lagging_Current_Power_Factor'])
data['Leading_Current_Power_Factor'] = pd.to_numeric(data['Leading_Current_Power_Factor'])
data['NSM'] = pd.to_numeric(data['NSM'])

# Usar datetime (tempo) com index dos dados do dataset
data = data.set_index("date")
data.index = pd.to_datetime(data.index,format='mixed') # Tive que adionar "format='mixed'" ,para retirar erro na linha 1152
# Converter Dados Categóricos em dados Numerais
#data['WeekStatus'].replace(['Weekday', 'Weekend'], [0, 1], inplace=True)
#data['Day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], [0, 1,2,3,4,5,6], inplace=True)
#data['Load_Type'].replace(['Light_Load', 'Medium_Load', 'Maximum_Load'], [0,1,2], inplace=True)
# Apaga dados faltantes
data = data.dropna()

# Printa = Exibi os nomes das colunas da tabela
#data.columns 
#data # Printa = Exibir a tabela na pagina
#data.shape #Exibir quantidade de colunas e linhas da tabela

#-----------------------------------------

# CORPO
## Cabeçalho: Título do trabalho e apresentação do pesquisador
st.title(':blue[Atividade 1 – Hands-On]')
st.markdown("""
            # 
            ---
            ## Visualização e Orientação por meio de dados
            """)
## Introdução: contém o objetivo/propósito da análise
st.markdown("""
             O presente trabalho tem o objetivo de explorar três tópicos de suma importância para tomadas de decisões a partir de ferramentas de software para Data Analytics:**
             - ETL – Extração, Tratamento e Carga de dados (Load) [Backend];
             - Aprendizado de Máquina (Classificação ou Previsão) [Engenharia de Dados];
             - Data Visualization [Frontend].
            ---
            Procedimentos para realização da atividade: 
            - Implemente cada um dos processos apresentados, ETL, aprendizado de máquina e Data Visualization, na seguinte sequência:
            - I. Carregue a base de dados no ambiente de desenvolvimento (Extração);
            - II. Analise cada variável individualmente e verifique dados faltantes e valores discrepantes;
            - III. Plote gráficos temporais de cada variável;
            - IV. Implemente modelos simples de aprendizado de máquina para realizar a predição do consumo energético. Utilize até dois modelos de aprendizado. Separe 75% dos dados obtidos para treinamento e 25% para avaliação dos modelos;
            - V. Avalie os resultados obtidos dos modelos de previsão com a métrica de R² Score. Essa métrica descreve o quão bem o modelo conseguiu generalizar o aprendizado em termos percentuais (%);
            - VI. Plote os resultados de previsão dos modelos e compare com os dados reais de saída do conjunto de testes.
            - VII. Gere um relatório em formato “pdf” dos experimentos realizados, analisando os dados obtidos com insights pessoais de cada time.
            ---
            """)


# SIDEBAR - Filtro de colunas
with st.sidebar:
    st.title('Filtros')
    cols_selected = \
        st.multiselect('Filtre os campos que deseja analisar:',
                       options=list(data.columns),
                       default=list(data.columns))

df_selected = data[cols_selected]

## SIDEBAR - Filtro de amostra
with st.sidebar:
    st.title('Amostra')
    amostra = \
        st.slider('Selecione a porcentagem da amostra desejada:', 
                  min_value=1, max_value=100, step=1,value=100)
   
    amostra = amostra/100
    df_selected = df_selected.sample(frac = amostra)

## CORPO - Carregando o dataset
st.header('Dataset')
    ## CORPO - Informações gerais do dataset
with st.expander('Dados gerais do dataset'):
    st.subheader('Primeiros registros')
    st.write(df_selected.head())
    
    st.subheader('Tamanho do Dataset')
    st.write('Quantidade de linhas:', df_selected.shape[0])
    st.write('Quantidade de colunas:', df_selected.shape[1])
    
    if st.checkbox('Exibir dados das colunas'):
        st.markdown("""
            - Date - Timestamp das amostras(DD/MM/YYYYHH;mm)
            - Usage_kWh - Consumo de Energia (kWh)
            - Lagging Power Factor- Fator de Potência Atrasado (%)
            - Usage_kWh - Consumo de Energia (kWh)
            - Lagging Power Factor - Fator de Potência Atrasado (%)
            - Lagging Current Reactive Power - Energia Reativa Atrasada (kVArh)
            - Leading Current Reactive Power - Energia Reativa Adiantada (kVArh)
            - CO2 - Emissão de CO² (ppm)
            - NSM - N° de Segundos a partir de 00:00
            - Week - Satus Fim de semana (0) ou dia da semana (1)
            - Day of week - Dia da semana
            - Load Type - Tipo de Consumo da Instalação - Light,Medium,Maximum
            """)

    st.subheader('Dados Faltantes')
    st.write(df_selected.isna().sum()[df_selected.isna().sum() > 0])

    st.subheader('Estatísticas Descritivas')
    st.write(df_selected.describe())

    #------------------------------------------------------------------

## CORPO - Análise Univariada
# Variáveis numéricas
st.header('Análise Univariada')
# st.subheader('Distribuição das variáveis numéricas')
with st.expander('Exibir ferramentas Graficas'):
    univar_campo_num =  \
        st.selectbox('Selecione o campo cuja distribuição você deseja avaliar:',
                    options=list(df_selected.select_dtypes(include=np.number)))

    st.write('Histograma')
    st.plotly_chart(px.histogram(
        data_frame=df_selected, 
        x=univar_campo_num, 
        text_auto=True,
        color_discrete_sequence=px.colors.qualitative.Set2))
        
    st.write('Box Plot')
    st.plotly_chart(px.box(data_frame=df_selected, y=univar_campo_num, color_discrete_sequence=px.colors.qualitative.Vivid))

    # Variáveis categóricas
    st.subheader('Participação das variáveis categóricas')
    univar_campo_cat =  \
        st.selectbox('Selecione o campo cuja distribuição você deseja avaliar:',
                    options=list(df_selected.select_dtypes(exclude=np.number)))

    st.write('Gráfico de pizza')
    contagem = data[univar_campo_cat].value_counts().values
    var_cat = data[univar_campo_cat].value_counts().index

    fig1, ax1 = plt.subplots(figsize = (5,3))
    ax1.pie(contagem, labels=var_cat, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

## CORPO - Análise Bivariada
st.header('Análise Bivariada')
with st.expander('Exibir ferramentas Graficas'):
    bivar_graf_option = \
        st.radio('Escolha um tipo de gráfico:',
                options=['Barras', 'Dispersão', 'Boxplot', 'Boxenplot'],
                key='bivar_graF_option')

    # Barras
    if bivar_graf_option == 'Barras':
        bivar_barras_cat =  \
            st.selectbox('Selecione uma variável numérica:',
                        options=list(df_selected.select_dtypes(include=np.number)))
            
        bivar_barras_num =  \
            st.selectbox('Selecione uma variável categórica:',
                        options=list(df_selected.select_dtypes(exclude=np.number)))
        st.write('Histograma')
        st.plotly_chart(px.histogram(
            data_frame=df_selected, 
            y=bivar_barras_cat, 
            x=bivar_barras_num,
            histfunc='avg',
            text_auto='.2f',
            category_orders={'WeekStatus': ['Weekday', 'Weekend'],
                            'Day_of_week': ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                            'Load_Type': ['Light_Load', 'Medium_Load', 'Maximum_Load']},
            color_discrete_sequence=px.colors.qualitative.Pastel1))

    ## Dispersão
    elif bivar_graf_option == 'Dispersão':
        bivar_dispersao_num1 =  \
            st.selectbox('Selecione a primeira variável numérica:',
                        options=list(df_selected.select_dtypes(include=np.number)))
            
        bivar_dispersao_num2 =  \
            st.selectbox('Selecione a segunda variável numérica:',
                        options=list(df_selected.select_dtypes(include=np.number)))
        st.write('Dispersão')
        if st.checkbox('Exibir linha de tendência'):
            st.plotly_chart(
                px.scatter(data_frame=df_selected, 
                        x=bivar_dispersao_num1, 
                        y=bivar_dispersao_num2,
                        trendline='ols',
                        trendline_color_override="red",
                        color_discrete_sequence=px.colors.qualitative.Antique))
        else:
            st.plotly_chart(
            px.scatter(data_frame=df_selected, 
                    x=bivar_dispersao_num1, 
                    y=bivar_dispersao_num2,
                    color_discrete_sequence=px.colors.qualitative.Antique)
        )
            
    ## Boxplot        
    elif bivar_graf_option == 'Boxplot':
        bivar_boxplot_num =  \
            st.selectbox('Selecione uma variável numérica:',
                        options=list(df_selected.select_dtypes(include=np.number)))
            
        bivar_boxplot_cat =  \
            st.selectbox('Selecione uma variável categórica:',
                        options=list(df_selected.select_dtypes(exclude=np.number)))
        st.write('Boxplot')
        st.plotly_chart(
            px.box(data_frame=df_selected, 
                    x=bivar_boxplot_cat, 
                    y=bivar_boxplot_num,
                        category_orders={'WeekStatus': ['Weekday', 'Weekend'],
                            'Day_of_week': ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                            'Load_Type': ['Light_Load', 'Medium_Load', 'Maximum_Load']})
        )

    # Boxenplot    
    elif bivar_graf_option == 'Boxenplot':
        bivar_boxenplot_num =  \
            st.selectbox('Selecione uma variável numérica:',
                        options=list(df_selected.select_dtypes(include=np.number)))
            
        bivar_boxenplot_cat =  \
            st.selectbox('Selecione uma variável categórica:',
                        options=list(df_selected.select_dtypes(exclude=np.number)))
        st.write('Boxenplot')
        fig2, ax2 = plt.subplots(figsize=(13,5))

        if bivar_boxenplot_cat == 'WeekStatus':
            sns.boxenplot(data=df_selected, 
                            x=bivar_boxenplot_cat, 
                            y=bivar_boxenplot_num,
                            order=['Weekday','Weekend'])

        elif bivar_boxenplot_cat == 'Day_of_week':
            sns.boxenplot(data=df_selected, 
                            x=bivar_boxenplot_cat, 
                            y=bivar_boxenplot_num,
                            order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

        elif bivar_boxenplot_cat == 'Load_Type':
            sns.boxenplot(data=df_selected, 
                            x=bivar_boxenplot_cat, 
                            y=bivar_boxenplot_num,
                            order=['Light_Load', 'Medium_Load', 'Maximum_Load'])

        else:
            sns.boxenplot(data=df_selected, 
                            x=bivar_boxenplot_cat, 
                            y=bivar_boxenplot_num)
    
        st.pyplot(fig2)

    #Pairplot
    else:
        bivar_pairplot = sns.pairplot(df_selected)
        st.pyplot(bivar_pairplot, key='bivar_pairplot')


## CORPO - Análise Multivariada
st.header('Análise Multivariada')
with st.expander('Exibir ferramentas Graficas'):
    multivar_graf_option = \
        st.radio('Escolha um tipo de gráfico:',
                options=['Barras', 'Dispersão', 'Boxplot', 'Violino', 'Pairplot'],
                key='multivar_graf_option')

    # Barras
    if multivar_graf_option == 'Barras':
        multivar_barras_num =  \
            st.selectbox('Selecione uma variável numérica:',
                        options=list(df_selected.select_dtypes(include=np.number)),
                        key='multivar_barras_num')
            
        multivar_barras_cat =  \
            st.selectbox('Selecione uma variável categórica:',
                        options=list(df_selected.select_dtypes(exclude=np.number)),
                        key='multivar_barras_cat')

        multivar_barras_seg =  \
            st.selectbox('Selecione uma variável categórica para segmentação:',
                        options=list(df_selected.select_dtypes(exclude=np.number)),
                        key='multivar_barras_seg')
        st.write('Barras') 
        fig3 = st.plotly_chart(px.histogram(data_frame=df_selected, 
                                            x=multivar_barras_cat, 
                                            y=multivar_barras_num,
                                            color=multivar_barras_seg,
                                            barmode='group',
                                            histfunc='avg',
                                            text_auto='.2f',
                                            category_orders={'WeekStatus': ['Weekday', 'Weekend'],
                                            'Day_of_week': ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                            'Load_Type': ['Light_Load', 'Medium_Load', 'Maximum_Load']},
                                            color_discrete_sequence=px.colors.qualitative.Pastel1))

    # Dispersão
    elif multivar_graf_option == 'Dispersão':
        multivar_campo_dispersao_1 =  \
            st.selectbox('Selecione primeira variável numérica:',
                        options=list(df_selected.select_dtypes(include=np.number)),
                        key = 'dispersao1_multivar')
            
        multivar_campo_dispersao_2 =  \
            st.selectbox('Selecione segunda variável numérica:',
                        options=list(df_selected.select_dtypes(include=np.number)),
                        key = 'dispersao2_multivar')
            
        multivar_campo_dispersao_3 =  \
            st.selectbox('Selecione uma variável categórica para segmentação:',
                        options=list(df_selected.select_dtypes(exclude=np.number)),
                        key = 'dispersao3_multivar')

        multivar_campo_dispersao_4 = \
            st.checkbox('Adicionar linha de tendência',
                        key = 'dispersao4_multivar')
        st.write('Disperção') 
        if multivar_campo_dispersao_4:
            st.plotly_chart( 
                px.scatter(data_frame=df_selected, 
                        x=multivar_campo_dispersao_1, 
                        y=multivar_campo_dispersao_2,
                        color=multivar_campo_dispersao_3,
                        trendline='ols',
                        color_discrete_sequence=px.colors.qualitative.Set1))

        else:
            st.plotly_chart( 
                px.scatter(data_frame=df_selected, 
                        x=multivar_campo_dispersao_1, 
                        y=multivar_campo_dispersao_2,
                        color=multivar_campo_dispersao_3,
                        color_discrete_sequence=px.colors.qualitative.Set1)
        )

    # Boxplot       
    elif multivar_graf_option == 'Boxplot':
        multivar_campo_boxplot_num =  \
            st.selectbox('Selecione uma variável numérica:',
                        options=list(df_selected.select_dtypes(include=np.number)),
                        key = 'boxplot1_multivar')
            
        multivar_campo_boxplot_cat =  \
            st.selectbox('Selecione uma variável categórica:',
                        options=list(df_selected.select_dtypes(exclude=np.number)),
                        key = 'boxplot2_multivar')
            
        multivar_campo_boxplot_seg =  \
            st.selectbox('Selecione uma variável categórica para segmentação:',
                        options=list(df_selected.select_dtypes(exclude=np.number)),
                        key = 'boxplot3_multivar')
        st.write('Boxplot') 
        st.plotly_chart(
            px.box(data_frame=df_selected, 
                    x=multivar_campo_boxplot_cat, 
                    y=multivar_campo_boxplot_num,
                    color=multivar_campo_boxplot_seg,
                    category_orders={'WeekStatus': ['Weekday', 'Weekend'],
                            'Day_of_week': ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                            'Load_Type': ['Light_Load', 'Medium_Load', 'Maximum_Load']})
        )

    # Violino       
    elif multivar_graf_option == 'Violino':
        multivar_campo_violin_num =  \
            st.selectbox('Selecione uma variável numérica:',
                        options=list(df_selected.select_dtypes(include=np.number)),
                        key = 'violin1_multivar')
            
        multivar_campo_violin_cat =  \
            st.selectbox('Selecione uma variável categórica:',
                        options=list(df_selected.select_dtypes(exclude=np.number)),
                        key = 'violin2_multivar')
            
        multivar_campo_violin_seg =  \
            st.selectbox('Selecione uma variável categórica para segmentação:',
                        options=list(df_selected.select_dtypes(exclude=np.number)),
                        key = 'violin3_multivar')

        fig3, ax3 = plt.subplots(figsize=(13,5))
        if multivar_campo_violin_cat == 'WeekStatus':
                sns.violinplot(data=df_selected, 
                    x=multivar_campo_violin_cat, 
                    y=multivar_campo_violin_num,
                    hue=multivar_campo_violin_seg,
                    order=['Weekday', 'Weekend'])

        elif multivar_campo_violin_cat == 'Day_of_week':
                sns.violinplot(data=df_selected, 
                    x=multivar_campo_violin_cat, 
                    y=multivar_campo_violin_num,
                    hue=multivar_campo_violin_seg,
                    order=['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])    
        
        elif multivar_campo_violin_cat == 'Load_Type':
                sns.violinplot(data=df_selected, 
                    x=multivar_campo_violin_cat, 
                    y=multivar_campo_violin_num,
                    hue=multivar_campo_violin_seg,
                    order=['Light_Load', 'Medium_Load', 'Maximum_Load'])    

        else:
            sns.violinplot(data=df_selected, 
                    x=multivar_campo_violin_cat, 
                    y=multivar_campo_violin_num,
                    hue=multivar_campo_violin_seg)
        st.write('Violino') 
        ax3.set_xlabel(multivar_campo_violin_cat, fontsize=20)
        ax3.set_ylabel(multivar_campo_violin_num, fontsize=20)
        ax3.tick_params(labelsize=13)
        sns.move_legend(ax3, "upper left", bbox_to_anchor=(1, 1))
        plt.setp(ax3.get_legend().get_texts(), fontsize='20') 
        plt.setp(ax3.get_legend().get_title(), fontsize='20') 
        st.pyplot(fig3)

    # Pairplot  
    else:
        multivar_campo_pairplot_seg =  \
            st.selectbox('Selecione uma variável categórica para segmentação:',
                        options=list(df_selected.select_dtypes(exclude=np.number)))

        multivar_pairplot = sns.pairplot(df_selected, hue = multivar_campo_pairplot_seg)
        st.pyplot(multivar_pairplot, key='multivar_pairplot')

## CORPO - Análise Data Visualization
st.header('Data Visualization')
with st.expander('Exibir ferramentas Graficas'):
    # Converter Dados Categóricos em dados Numerais
    data['WeekStatus'].replace(['Weekday', 'Weekend'], [0, 1], inplace=True)
    data['Day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], [0, 1,2,3,4,5,6], inplace=True)
    data['Load_Type'].replace(['Light_Load', 'Medium_Load', 'Maximum_Load'], [0,1,2], inplace=True)
    # Apaga dados faltantes

    Dvisualization_graf_option = \
        st.radio('Escolha um tipo de gráfico:',
                options=['Grafico_Correlação', 'Grafico_Histograma', 'Grafico_Boxplot'],
                key='Dvisualization_graf_option')

    # Correlação
    if Dvisualization_graf_option == 'Grafico_Correlação':
        st.write('Grafico de Correlação')      
        hor_size = 12
        ver_size = 12
        plt.subplots(figsize=(hor_size,ver_size))
        fig0, (ax1) = plt.subplots(figsize=(hor_size,ver_size))
        correlation_mat = data.corr()
        sns.heatmap(correlation_mat, annot = True) # Usa a biblioteca seaborn
        plt.figure(figsize=(10,5))
        plt.rcParams['font.size'] = '9'
        #plt.show()
        st.pyplot(fig0)

    ## Consumo de Energia ao Longo do Tempo
    elif Dvisualization_graf_option == 'Grafico_Histograma':
        # Dvisualization_Histograma_num1 =  \
        #     st.selectbox('Selecione a primeira variável numérica:',
        #                  options=list(df_selected.select_dtypes(include=np.number)),
        #                  key='Dvisualization_Histograma_num1')
        # data_inicio = st.date_input("Data Inicial", value=None)
        # data_fim = st.date_input("Data Final", value=None)
        # st.title(':blue[Visulização do Consumo de Energia ao Longo do Tempo]')
        st.write('Grafico do Consumo de Energia ao Longo do Tempo')  
        hor_size = 20
        ver_size = 5
        fig1, (ax1) = plt.subplots(figsize=(hor_size,ver_size))
        data['Usage_kWh'].plot(ax = ax1, label = 'Energia Consumida (kWh)', style = '-', color = 'red')
        plt.xlim(pd.to_datetime("2018-01"),pd.to_datetime("2018-07-15"))
        ax1.set_title("Dados temporais no eixo X")  # Add a title to the axes.
        ax1.set_xlabel('Data') # Add an x-label to the axes.
        ax1.set_ylabel('Consumo em KwW') # Add an y-label to the axes.
        # plt.xlim(pd.to_datetime("data_inicio"),pd.to_datetime("data_fim"))
        ax1.legend()
        st.pyplot(fig1)

    # Boxplot
    elif Dvisualization_graf_option == 'Grafico_Boxplot':
        # Dvisualization_Boxplot_num1 =  \
        #     st.selectbox('Selecione a primeira variável numérica:',
        #                  options=list(df_selected.select_dtypes(include=np.number)),
        #                  key='Dvisualization_Boxplot_num1')
        st.write('Boxplot')  
        sns.boxplot(x=data['Usage_kWh'])





## CORPO - machine learning
st.header('Machine Learning')
with st.expander('Exibir ferramentas Graficas'):
    # Dividir conjuntos para treinamento e testes
    X = data.drop('Usage_kWh', axis = 1)
    y = data['Usage_kWh']
    X = X.values
    y = y.values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10, shuffle = False)
    #--------------------------------------------

    Machine_Learning_graf_option = \
        st.radio('Escolha um tipo de gráfico:',
                options=['Treino', 'Teste'],
                key='Machine_Learning_graf_option')


    # Teste
    if Machine_Learning_graf_option == 'Teste':
        st.write('Grafico de Correlação')
        # Visualização do Conjunto de Dados e Treino (PS: Os gráficos estão sem dados temporais no eixo "X")
        fig, ax = plt.subplots(figsize=(20,5))
        plt.plot(y_test, label = 'Test Set', color = 'red')
        ax.set_title("Dados temporais no eixo X")  # Add a title to the axes.
        ax.set_xlabel('Amostragem') # Add an x-label to the axes.
        ax.set_ylabel('Consumo em KwW') # Add an y-label to the axes.
        ax.legend(['Test Set'])
    
        st.pyplot(fig)
        # plt.show()      
    

    ## Consumo de Energia ao Longo do Tempo
    elif Machine_Learning_graf_option == 'Treino':
    
        fig, ax = plt.subplots(figsize=(15,5))
        plt.plot(y_train, label = 'Training Set', color = 'red')        
        ax.set_title("Dados temporais no eixo X")  # Add a title to the axes.
        ax.set_xlabel('Amostragem') # Add an x-label to the axes.
        ax.set_ylabel('Consumo em KwW') # Add an y-label to the axes.
        ax.legend(['Training Set'])
        st.pyplot(fig)
        # plt.show()

    # Implementação do Modelo de Aprendizado
    # Regressão Linear
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(x_train, y_train)

    # Predições
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)


    ## CORPO - Avaliação dos Resultados Preditos
    st.header('Resultados Preditos')
    Preditos_graf_option = \
        st.radio('Escolha um tipo de gráfico:',
                options=['Resultado_Treino', 'Resultado_Teste'],
                key='Preditos_graf_option')


    # Teste
    if Preditos_graf_option == 'Resultado_Treino':
        # Avaliação dos Resultados Preditos (PS: Os gráficos estão sem dados temporais no eixo "X")
        # Avaliação dos Resultados de Treino
        st.write('Avaliação dos Resultados de Treino')
        # plt.figure(figsize=(25,6))
        fig, ax = plt.subplots(figsize=(25,6))
        plt.plot(y_train, color = 'red', linewidth=2.0, alpha = 0.6)
        plt.plot(y_train_pred, color = 'blue', linewidth=0.8)
        plt.legend(['Atual','Predito'])
        plt.title("Training data prediction")
        ax.set_xlabel('Amostragem') # Add an x-label to the axes.
        ax.set_ylabel('Consumo em KwW') # Add an y-label to the axes.
        st.pyplot(fig)
        # plt.show()
        score = r2_score(y_train, y_train_pred)
        # print(f'Resultado de R² para o conjunto treino: {score*100: 0.4f}') 
        st.write(f'Resultado de R² para o conjunto testes: {score*100: 0.4f}')    
    

    ## Consumo de Energia ao Longo do Tempo
    elif Preditos_graf_option == 'Resultado_Teste':
        # Avaliação dos Resultados de Testes
        st.write('Avaliação dos Resultados de Testes')
        # plt.figure(figsize=(25,6))
        fig, ax = plt.subplots(figsize=(25,6))
        plt.plot(y_test, color = 'red', linewidth=2.0, alpha = 0.6)
        plt.plot(y_test_pred, color = 'blue', linewidth=0.8)
        plt.legend(['Atual','Predito'])
        plt.title("Test data prediction")
        ax.set_xlabel('Amostragem') # Add an x-label to the axes.
        ax.set_ylabel('Consumo em KwW') # Add an y-label to the axes.
        st.pyplot(fig)
        # plt.show()
        score = r2_score(y_test, y_test_pred)
        # print(f'Resultado de R² para o conjunto testes: {score*100: 0.4f}')
        st.write(f'Resultado de R² para o conjunto testes: {score*100: 0.4f}')
   





















