from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#import streamlit as st

app = Flask('stroke_predict')


#logistic regression model
model_file_path = 'app/models/lr_model_stroke_prediction.sav'
model = pickle.load(open(model_file_path, 'rb'))

#encoding model
encoding_model_filepath = 'app/models/encoding_model_stroke_prediction.sav'
encoding_model = pickle.load(open(encoding_model_filepath, 'rb'))

# global variables for prediction model

# ML section start:
numerical = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
categorical = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
le_enc_cols = ['ever_married']
y_n_map = {'yes':1, 'no': 0}
treshold = 0.12

@app.route('/stroke_predict', methods = ['POST'])
def stroke_predict():
    customer = request.get_json()
    df_input = pd.DataFrame(customer)
    scaler = MinMaxScaler()
    
    df_input[numerical]=scaler.fit_transform(df_input[numerical])
    
    for col in le_enc_cols:
        df_input[col] = df_input[col].map(y_n_map)
    
    dict_df = df_input[categorical+numerical].to_dict(orient = 'records')
    X = encoding_model.transform(dict_df)
    y_pred_proba = model.predict_proba(X)[:,1]
    stroke_decision = (y_pred_proba>=treshold)
    
    result = {"stroke_probability":float(y_pred_proba[0]),"stroke_risk": bool(stroke_decision[0])}
    

if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

