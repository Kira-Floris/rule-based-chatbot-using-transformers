import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/Kira-Floris/chatbot-hackathon/master/data.csv')

text_df = df[df['Document'].apply(lambda x: len(x)>100)]

unique_text_df = text_df.drop_duplicates(subset='Document', keep='last')

import nltk
nltk.download('stopwords')
from Questgen import main

qe = main.BoolQGen()

import json

def generate_questions(df):
  questions = []
  labels = []
  intents = {'intents':[]}
  for index, row in df.iterrows():
    document = row[-1]
    temp_intent = {}
    temp = document.split('.')
    temp_questions = []
    for item in temp:
      payload = {
        'input_text':text
      }
      q = qe.predict_boolq(payload)['Boolean Questions']
      temp_questions.extend(q)
    labels.extend([index]*len(temp_questions))
    temp_intent['tag'] = index
    temp_intent['responses'] = temp 
    intents['intents'].append(temp_intent)
    questions.extend(temp_questions)
  with open('intents.json', 'w+') as f:
    intents = json.dumps(intents)
    f.write(intents)
  with open('df.json', 'w+') as f:
    questions = json.dumps({'question':questions,'label':labels})
    f.write(questions)
  return intents, questions

x = generate_questions(unique_text_df)