import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle

st.set_page_config(
    page_title='Stroke_Prediction_App',
    #page_icon = ""
    initial_sidebar_state = "expanded",
)

# session state initialization
if 'df_input' not in st.session_state:
     st.session_state['df_input'] = pd.DataFrame()


if 'model' not in st.session_state:
     st.session_state['model'] = None

if 'encoding_model' not in st.session_state:
     st.session_state['encoding_model'] = None

# ML section start:
numerical = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
categorical = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
le_enc_cols = ['ever_married']
y_n_map = {'yes':1, 'no': 0}

#logistic regression model
model_file_path = 'D:\BU AI\Logistic regression project\stroke_prediction_streamlit_app\models\lr_model_stroke_prediction.sav'
st.session_state['model'] = pickle.load(open(model_file_path, 'rb'))

#encoding model
encoding_model_filepath = 'D:\BU AI\Logistic regression project\stroke_prediction_streamlit_app\models\encoding_model_stroke_prediction.sav'
st.session_state['encoding_model'] = pickle.load(open(encoding_model_filepath, 'rb'))

@st.cache_data
def predict_stroke_minmax(df_input, treshold):
    scaler = MinMaxScaler()
    
    df_original = df_input.copy()
    df_input[numerical]=scaler.fit_transform(df_input[numerical])
    
    for col in le_enc_cols:
        df_input[col] = df_input[col].map(y_n_map)
    
    dict_df = df_input[categorical+numerical].to_dict(orient = 'records')
    X = st.session_state['encoding_model'].transform(dict_df)
    y_pred_proba = st.session_state['model'].predict_proba(X)[:,1]
    stroke_decision = (y_pred_proba>=treshold).astype(int)
    df_original['stroke decision'] = stroke_decision
    df_original['stroke predicted probability'] = y_pred_proba
    return df_original



df_predicted = pd.DataFrame()

#sidebar: 
with st.sidebar:
    st.title('Ввод данных')

    tab1, tab2 = st.tabs(["Load file", "Add information manually"])
    with tab1:
        st.header('Load file')
        uploaded_file = st.file_uploader('Выбрать csv файл', type = ['csv','xlsx'])
        if uploaded_file is not None:
            treshold = st.slider('Порог вероятности инсульта',0.0, 1.0, 0.5)
            prediction_button = st.button('Предсказать', type ='secondary')
            st.session_state['df_input'] = pd.read_csv(uploaded_file)
            if prediction_button:
                 df_predicted = predict_stroke_minmax(st.session_state['df_input'], treshold)
            #df = pd.read_csv(uploaded_file)
            #st.write(df)

    with tab2:
        st.header('Fill the query')
    #st.write('Hello world!')

## sidebar section ends here

st.image('https://thumbs.dreamstime.com/z/human-brain-analyzed-magnifying-glass-cogwheels-working-inside-isolated-human-brain-analyzed-magnifying-glass-cogwheels-121321631.jpg?w=992', width = 400)

st.title('Welcome to Stroke Prediction App')

## expander section:
with st.expander("О проекте"):
    st.write('Данное приложение предсказывает вероятность инсульта у индивидуума. Данные используются из открытых источников. На основе анкетных данных модель предсказывает риск инсульта')

if len(st.session_state['df_input'])>0:
    st.subheader('Данные из файла')
    st.write(st.session_state['df_input'])

if len(df_predicted)>0:
    st.subheader('Результаты прогноза по инсульту')
    st.write(df_predicted)
     
#st.write('heyyyyy!')