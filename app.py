# %%writefile app.py
# setup
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import streamlit as st
from streamlit_chat import message
import detectlanguage
import translators as ts
import translators.server as tss
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import json
from transformers import RobertaTokenizer, RobertaModel
import re
import requests

# tools setup
# setup detect language api keys
detectlanguage.configuration.api_key = '7a1c7069f905116a159438796c09db8e'

# setup sbert model
st.session_state['sbert_model'] = SentenceTransformer('all-MiniLM-L6-v2')

# streamlit setup
st.set_page_config(
    page_title='Chatbot',
    page_icon=':robot:'
)
st.header('Chat With Us')
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []
    

# verify answer
def verify_answer(question, answer):
    qs = [question]
    ans = [answer]
    
    embeddings1 = st.session_state['sbert_model'].encode(qs, convert_to_tensor=True)
    embeddings2 = st.session_state['sbert_model'].encode(ans, convert_to_tensor=True)
    
    score = util.cos_sim(embeddings1, embeddings2)
    return score[0]

# translation function or class
class Translation:
    def __init__(self, text, data_lang='en'):
        self.text = text
        self.language = detectlanguage.detect(str(text))[0]['language']
        self.data_lang = data_lang

    def encode(self):
        if self.language == self.data_lang:
            return self.text
        else:
            translation = tss.google(self.text, self.language, self.data_lang)
            return translation
        
    def bot_response(self, title, confidence, link):
        confidence_text = ''
        if confidence>0.75:
            confidence_text = 'You can find information related to "{}" on this link: {}'.format(title, link)
        elif confidence<=0.75 and confidence>0.50:
            confidence_text = '50-50 chance you will find information related to "{}" on this link: {}'.format(title, link)
        elif confidence<=0.50 and confidence>0.30:
            confidence_text = 'Am not sure, but you might find information related to "{}" on this link: {}'.format(title, link) 
        else:
            confidence_text = "Sorry, I couldn't find information. Can you elaborate more on the question?"
        return confidence_text

    def decode(self, title, link):
        confidence = verify_answer(self.text, title)
        if self.language == self.data_lang:
            return self.bot_response(title, confidence, link)
        else:
            answer = self.bot_response(title, confidence, link)
            translation = tss.google(answer, self.data_lang, self.language)
            return translation
        

# display user and bot chat from dictionary
# keep scrolling up as input
# input and answer saved in a dictionary 
def get_text():
    user_text = st.text_input('You: ', placeholder='Message', key='input')
    return user_text

def get_response(text, link):
    obj = {'text':str(text)}
    try:
        ans = requests.post(link, json=obj)
        response = json.loads(ans.text)
        return response['link'], response['title']
    except Exception as err:
        st.write(err)
        return

user_text = get_text()

if user_text:
    trans = Translation(user_text)
    user_text_translated = trans.encode()
    st.write(user_text_translated)
    
    bot_answer = get_response(user_text_translated, 'http://40d5-35-229-75-39.ngrok.io/')
    st.write(bot_answer)
    bot_text_translated = trans.decode(bot_answer[0],bot_answer[1])
    
    st.session_state.past.append(user_text)
    st.session_state.generated.append(bot_text_translated)
    
# display chat messages
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
