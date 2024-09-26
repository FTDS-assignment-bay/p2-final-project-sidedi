import streamlit as st
import eda, predict

st.set_page_config(page_title= 'Indonesia Online Gambling Website Prediction', layout = 'wide', initial_sidebar_state= 'expanded')


st.sidebar.title("Navigation")
navigation = st.sidebar.selectbox(label = 'Go to', 
                                  options = ["EDA", "PREDICTION"])

if navigation == 'EDA':
    eda.run()
else:
    predict.run()