# Import Library
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import streamlit as st

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.metrics import Precision


# Download NLTK package
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('all')
nltk.download('punkt_tab')

# Define the custom precision metric
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_value = true_positives / (predicted_positives + K.epsilon())
    return precision_value

# Load the model with the custom precision metric
model = load_model('model_final_FP', custom_objects={'precision': precision})



def run():
    st.title('Sidedi - Sistem Pendeteksi Website Judi')
    st.image('https://pontianakinfo.disway.id/upload/cec9a564c5afe35746699bf4c08a5345.png', width=500, caption='Identifying Online Gambling Website')


    st.sidebar.title('Input Data to Predict')
    st.sidebar.title('About this page')
    st.sidebar.write('This is page for predicting gambling website')

    # Membuat form untuk input data
    st.write("Input Content/Data from the website for Prediction:")


    # Form input data
    with st.form(key = 'content'):
        Content_Input = st.text_input('Write content', value = '')
        

        #Submit button
        submit_button = st.form_submit_button(label = 'Predict')

    # Additional stopwords
    list_stopwds = ['menu', 'close', 'home', 'pemilu', 'Pendidikan', 'advertisement']

    # Define stopword for indonesian words
    stpwds_id = list(set(stopwords.words('indonesian')))

    # Append the stopword with the additional initialized stopwords
    stpwds_id = stpwds_id + list_stopwds


    # Define Stemmer
    stemmer = StemmerFactory().create_stemmer()

    # Function for text processing
    def text_preprocessing(text):
        '''
        Function to preprocess text including case folding, mention removal, hashtag removal,
        newline removal, whitespace removal, url removal, non-letter removal, tokenization,
        stopword removal, and stemming
        '''
        # Case folding
        text = text.lower()

        # URL removal
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"www.\S+", " ", text)

        # Mention removal
        text = re.sub("@[A-Za-z0-9_]+", " ", text)

        # Hashtags removal
        text = re.sub("#[A-Za-z0-9_]+", " ", text)

        # Newline removal (\n)
        text = re.sub(r"\\n", " ",text)

        # Remove navigation and irrelevant sections
        text = re.sub(r'(?i)(menu close|edisi|daftar isi|newsletter|penulis|mitra|privasi|syarat).*?$', '', text, flags=re.DOTALL)

        # Remove promotional phrases and repetitive categories
        text = re.sub(r'(INTERNAL FEEDBACK|Slot Gacor|Situs Judi Slot|Download Aplikasinya|Belanja Sekarang|Vouchernya|Produk Eksklusif|Hubungi kami di Live Chat).*?(\.\s|$)', '', text, flags=re.DOTALL)

        # Whitespace removal
        text = text.strip()

        # Remove non-latin words from the sentences.
        text = re.sub(r'[^\x00-\x7f]', r'', text)

        # Non-letter removal (such as emoticon, symbol (like μ, $, 兀), etc
        text = re.sub("[^A-Za-z\s']", " ", text)

        # Tokenization
        tokens = word_tokenize(text)

        # Stopwords removal
        tokens = [word for word in tokens if word not in stpwds_id]

        # Removing Words less than 2 lengths
        tokens =  [word for word in tokens if len(word) > 2]

        # Stemming
        tokens = [stemmer.stem(word) for word in tokens]

        # Combining Tokens
        text = ' '.join(tokens)

        return text

    if submit_button:
        Content = {
        'content': Content_Input
        }

        df = pd.DataFrame([Content])

        st.write("## Data Summary")
        st.dataframe(df)

        # Apply the test processing function
        df['text_processed'] = df['content'].apply(lambda x: text_preprocessing(x))
        df

        # predict
        prediction = np.argmax(model.predict(df['text_processed']), axis=1)

        st.write("## Data Prediction")

        if prediction[0] == 0:
            st.write(f'Categorized as NON GAMBLING WEBSITE')
            st.image('https://media.licdn.com/dms/image/C4E12AQEthJyiP6x4SA/article-cover_image-shrink_600_2000/0/1520080512068?e=2147483647&v=beta&t=pK-YUWRqct7VaJ8vDeyWtqbB8uEWArSkxJrR5nkp6MM', width=500, caption='Secure Web')
            

        else:
            st.write(f'Categorized as GAMBLING WEBSITE')
            st.write(f'Please take action!')
            st.image('https://t4.ftcdn.net/jpg/01/52/59/55/360_F_152595566_rzcoccuRYa4jekwWznYAPNhIOtltPae7.jpg', width=500, caption='Block!')
