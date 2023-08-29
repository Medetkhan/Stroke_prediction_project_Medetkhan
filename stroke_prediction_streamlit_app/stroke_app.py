import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
import plotly.express as px

st.set_page_config(
    page_title='Stroke_Prediction_App',
    #page_icon = ""
    initial_sidebar_state = "expanded",
)

# session state initialization
if 'df_input' not in st.session_state:
     st.session_state['df_input'] = pd.DataFrame()

if 'df_predicted' not in st.session_state:
    st.session_state['df_predicted'] = pd.DataFrame()

if 'tab_selected' not in st.session_state:
    st.session_state['tab_selected'] = None


if 'model' not in st.session_state:
     st.session_state['model'] = None

if 'encoding_model' not in st.session_state:
     st.session_state['encoding_model'] = None

def reset_session_state():
    st.session_state['df_input'] = pd.DataFrame()
    st.session_state['df_predicted'] = pd.DataFrame()


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
    df_original['stroke_decision'] = stroke_decision
    df_original['stroke_predicted_probability'] = y_pred_proba
    return df_original



#df_predicted = pd.DataFrame()
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

#sidebar: 
with st.sidebar:
    st.title('Ввод данных')

    tab1, tab2 = st.tabs(["Load file", "Add information manually"])
    with tab1:
        
        st.header('Load file')
        uploaded_file = st.file_uploader('Выбрать csv файл', type = ['csv','xlsx'], on_change = reset_session_state)
        if uploaded_file is not None:
            treshold = st.slider('Порог вероятности инсульта',0.0, 1.0, 0.5, key='slider1')
            prediction_button = st.button('Предсказать', type ='secondary')
            st.session_state['df_input'] = pd.read_csv(uploaded_file)
            if prediction_button:
                 st.session_state['df_predicted'] = predict_stroke_minmax(st.session_state['df_input'], treshold)
                 st.session_state['tab_selected'] = 'tab1'
            #df = pd.read_csv(uploaded_file)
            #st.write(df)

    with tab2:
        
        st.header('Fill the query')
        patient_id = st.text_input('id', placeholder = '000000', help = 'Введите ID')
        gender = st.selectbox('Gender', ('female','male'))
        age = st.text_input('age', '00')
        hypertension = st.selectbox('hypertension', ('0','1'), help = 'гипертония если есть то нажмите 1, в противном случае 0')
        heart_disease = st.selectbox('heart disease', ('0','1'), help = 'Имеются ли болезни сердца')
        ever_married = st.selectbox('marriage status', ('yes', 'no'))
        work_type = st.selectbox('work type', ('private', 'self employed', 'govt job'))
        residence_type = st.selectbox('residence type', ('urban', 'rural'))
        avg_glucose_level = st.text_input('average glucose level', placeholder = '000.00', help = 'уровень глюкозы')
        bmi = st.text_input('bmi', placeholder = '00.00')
        smoking_status = st.selectbox('smoking status', ('formerly smoked', 'never smoked', 'smokes'))

        if patient_id !='':
            treshold = st.slider('Порог вероятности инсульта', 0.0, 1.0, 0.5, key='slider2')
            prediction_button_tab2 = st.button('Предсказать', type ='primary', use_container_width=True, key = 'button2')

            if prediction_button_tab2:
                st.session_state['tab_selected'] = 'tab2'
                st.session_state['df_input']= pd.DataFrame({
                    'patient_id': patient_id,
                    'gender': gender,
                    'age': age,
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                    'ever_married': ever_married,
                    'work_type': work_type,
                    'residence_type': residence_type,
                    'avg_glucose_level': avg_glucose_level,
                    'bmi': bmi,
                    'smoking_status': smoking_status
                }, index=[0]) #index=[0] 
                st.session_state['df_predicted'] = predict_stroke_minmax(st.session_state['df_input'], treshold)




    #st.write('Hello world!')

## sidebar section ends here

st.image('https://thumbs.dreamstime.com/z/human-brain-analyzed-magnifying-glass-cogwheels-working-inside-isolated-human-brain-analyzed-magnifying-glass-cogwheels-121321631.jpg?w=992', width = 400)

st.title('Welcome to Stroke Prediction App')

## expander section:
with st.expander("О проекте"):
    st.write('Данное приложение предсказывает вероятность инсульта у индивидуума. Данные используются из открытых источников. На основе анкетных данных модель предсказывает риск инсульта')

if len(st.session_state['df_input'])>=0:
    if len(st.session_state['df_input'])==0:
        st.subheader('Данные из файла')
        st.write(st.session_state['df_input'])
    else:
        with st.expander('Входные данные'):
            st.write(st.session_state['df_input'])
    #st.line_chart(st.session_state['df_input'][['age', 'bmi']])

if len(st.session_state['df_predicted'])>0 and st.session_state['tab_selected']=='tab2':
    if st.session_state['df_predicted']['stroke_decision'][0]==0:
        st.subheader(f"has :blue[no] stroke risks with following probabilty {(1-st.session_state['df_predicted']['stroke_predicted_probability'][0])*100:.2f}%")
    else:
        st.subheader(f"Has :red[some] stroke risks with following probability {(st.session_state['df_predicted']['stroke_predicted_probability'][0])*100:.2f}%")


if len(st.session_state['df_predicted'])>0 and st.session_state['tab_selected']=='tab1':
    st.subheader('Результаты прогноза по инсульту')
    st.write(st.session_state['df_predicted'])
    res_csv = convert_df(st.session_state['df_predicted'])
    st.download_button(
        label = 'Download all predictions',
        data = res_csv,
        file_name = 'df_predicted_stroke.csv',
        mime = 'text/csv',
    )

    #fig = px.histogram(st.session_state['df_predicted'], x = 'stroke decision', color = 'stroke decision')
    #st.plotly_chart(fig, use_container_width = True)

    risk_of_stroke = st.session_state['df_predicted'][st.session_state['df_predicted']['stroke_decision']==0]

    if len(risk_of_stroke) > 1:
        st.subheader('Люди с высоким риском инсульта')
        st.write(risk_of_stroke)

        res_risky_csv = convert_df(risk_of_stroke)
        st.download_button(
            label = 'Download data with stroke positive predictions',
            data = res_risky_csv,
            file_name = 'df_positive_stroke_risk.csv',
            mime = 'text/csv',
        )

#st.write('heyyyyy!')