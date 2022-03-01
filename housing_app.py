import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import joblib

st.title("Housing Price Calculator")
st.write("A machine learning app to calculate housing price")
st.sidebar.title("Build your own house")
st.sidebar.write("Use the slider to calculate the price")
bedrooms = st.sidebar.slider ('Bedrooms',0,5,1)
bathrooms = st.sidebar.slider ('Bathrooms',0,4,1)
floors = st.sidebar.slider ('Floors',1,3,1)

st.write("Please pay")
filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)
prediction = round(loaded_model.predict([[bedrooms, bathrooms, floors]])[0])
st.write(f"Estimated price is:") 
st.subheader(prediction)