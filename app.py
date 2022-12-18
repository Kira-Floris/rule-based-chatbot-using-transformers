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

# tools setup
# setup detect language api keys
detectlanguage.configuration.api_key = '7a1c7069f905116a159438796c09db8e'

# setup sbert model
st.session_state['sbert_model'] = SentenceTransformer('all-MiniLM-L6-v2')

# device setup
st.session_state['device'] = torch.device('cuda')

# bert and tokenizer setup
st.session_state['tokenizer'] = RobertaTokenizer.from_pretrained('roberta-base')
st.session_state['bert'] = RobertaModel.from_pretrained('roberta-base') 

# setup LabelEncoder
le = LabelEncoder()

# sequence length
max_seq_len = 16

# load intents.json with cache
f = open('intents.json')
st.session_state['intents'] = json.load(f)

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
    

# model class
# model
class BERT_Arch(nn.Module):
  def __init__(self, bert=st.session_state['bert'], dropout=0.2, hl1=768, hl2=512, hl3=256, out=len(st.session_state['intents']['intents'])):
    super(BERT_Arch, self).__init__()
    self.bert = bert
    self.dropout = nn.Dropout(dropout)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(hl1, hl2)
    self.fc2 = nn.Linear(hl2, hl3)
    self.fc4 = nn.Linear(hl3, out)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, sent_id, mask):
    cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]
    x = self.fc1(cls_hs)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc3(x)
    x = self.softmax(x)
    return x

# load model with cache
model_path = 'model.pth'
st.session_state['model'] = torch.load(model_path)

# label encoder
questions = pd.read_csv('questions.csv')
le = LabelEncoder()
questions['label'] = le.fit_transform(questions['label'])


# prediction function
class Prediction:
    def __init__(self, model=st.session_state['model'], tokenizer=st.session_state['tokenizer']):
        self.model = model
        self.model_eval = self.model.eval()
        self.tokenizer = tokenizer
    
    def __call__(self, text):
        text = re.sub(r'[^a-zA-Z ]+', '', text)
        test_text = [text]
        
        tokens_test_data = self.tokenizer(
                test_text,
                max_length = max_seq_len,
                pad_to_max_length=True,
                truncation=True,
                return_token_type_ids=False
            )
        
        test_seq = torch.tensor(tokens_test_data['input_ids'])
        test_mask = torch.tensor(tokens_test_data['attention_mask'])
        
        preds = None
        with torch.no_grad():
            preds = self.model(
                test_seq.to(
                    st.session_state['device']), 
                test_mask.to(st.session_state['device'])
                )
        preds = preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis = 1)
        # print('Intent Identified: ', le.inverse_transform(preds)[0])
        intent = le.inverse_transform(preds)[0]
        
        def get_response(intent):
            for i in st.session_state['intents']['intents']:
                if i['tag'] == intent:
                    result = i['responses']
                    link = i['tag']
                    break
            return result, link
        return get_response(intent) 
        

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

user_text = get_text()

if user_text:
    trans = Translation(user_text)
    prediction = Prediction()
    user_text_translated = trans.encode()
    
    bot_answer = prediction(user_text_translated)
    bot_text_translated = trans.decode(bot_answer[0],bot_answer[1])
    
    st.session_state.past.append(user_text)
    st.session_state.generated.append(bot_text_translated)
    
# display chat messages
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
