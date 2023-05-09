import os
import streamlit as st
import numpy as np
import pickle
from pickle import load
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = load(open(r"models\std.pkl", 'rb'))
lr = load(open(r"models\lr_model.pkl", 'rb'))

st.title(":Blue[Diabetes Prediction]")

glucose = st.slider(":Red[Select Glucose Level]", 0, 200, 100)

blood_pressure = st.slider(":Red[Select Blood Pressure Level]", 0, 122, 62)

skin_thickness = st.slider(":Red[Select Skin Thickness]", 0, 100, 20)

insulin = st.slider(":Red[Select Insulin Level]", 0, 900, 150)

bmi = st.slider(":Red[Select BMI]", 0, 70, 25)

dpf = st.slider(":Red[Select Diabetes Pedigree Function]", 0.000, 2.500, 0.500)

age = st.slider(":Red[Select Your Age]", 21, 81, 30)

if st.button('Predict'):
    query_point = np.array([glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
    query_point = query_point.reshape(1, -1)
    query_point_transformed = scaler.transform(query_point)
    prediction = lr.predict(query_point_transformed)
    if prediction == 0:
        st.success("You don't have Diabetes 😊!")
        st.image("images.jpeg")
    else:
        st.error("You have Diabetes 😥!")
        st.image("sad diabetes.jpg")
