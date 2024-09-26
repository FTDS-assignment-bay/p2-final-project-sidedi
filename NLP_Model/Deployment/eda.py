import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from PIL import Image
from io import BytesIO

def run(): # Agar ga langsung muncul pada saat disatuin di satu app

    st.sidebar.title('Sidedi - Sistem Pendeteksi Website Judi')
    st.sidebar.title('About this page')
    st.sidebar.write('This page describes the online gambling website in Indonesia')


    st.title('Online Gambling Website Indonesia Identification')
    st.title('Sidedi - Sistem Pendeteksi Website Judi')

    st.image('https://blue.kumparan.com/image/upload/fl_progressive,fl_lossy,c_fill,q_auto:best,w_1280/v1634025439/01hx6qd6hb24xracth6wbeqz7j.jpg', width=800, caption='Online Gambling Website')

    st.write('## Data Information:')
  
    # Function to fetch and display image
    def load_image_from_drive(file_id):
        url = f'https://drive.google.com/uc?id={file_id}'
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img

    # File IDs from Google Drive
    file_ids = {
        "WordCloud Gambling Website": "1EoDmCMlg-in0T-j0fU5pJdGeInRI8KXB",
        "WordCloud Non Gambling / News Related Website": "1vpmv1-r66ysxdhM9gGy6D0peg14ZZynn"
    }

    # Display images in Streamlit
    for title, file_id in file_ids.items():
        st.subheader(f'**{title}**')
        img = load_image_from_drive(file_id)
        st.image(img, width=800) 