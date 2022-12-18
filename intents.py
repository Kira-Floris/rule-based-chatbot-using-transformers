# quesgen setup and nltk
import nltk
nltk.download('stopwords')
from Questgen import main
import pandas as pd
import json

qe = main.BoolQGen()

# quesgen
def questgen_generate(answer, n=2, qe=qe):
  payload = {
      'input_text':str(answer)
  }
  output = qe.predict_boolq(payload)
  return output['Boolean Questions'][:n]

import json
import pandas as pd

question = """Guide me to {}?
Where do you get {}?
Where can I find information about {}?
What is the link to the {}?
How do you get {}?
What can you tell me about {}?"""

def generate_intents_and_df(df, 
                            qe_n=45, 
                            intents_json='intents.json', 
                            questions_df='questions.csv'):
  
  # generate intents
  intents = {
      'intents':[]
  }
  questions_df_ls = [] 
  labels_df_ls = []
  for index, row in df.iterrows():
    intent = {}
    intent['tag'] = row[1]
    intent['response'] = row[0]
    questions = []
    for i in range(len(question.split('\n'))):
      questions.append(question.split('\n')[i].format(intent['response']))
    # generate more questions using questgen
    # string longer than 45 brings accurate questions
    if len(str(intent['tag']))>qe_n:
      questions.extend(questgen_generate(intent['tag']))
    else:
      pass
    intents['intents'].append(intent)
    labels_df_ls.extend([intent['tag']] * len(questions))
    questions_df_ls.extend(questions)

  # save intents into json file
  with open(intents_json, 'w+') as f:
    intents = json.dumps(intents)
    f.write(intents)
  # save questions and intent tag as labels into csv file 
  df = pd.DataFrame()
  df['text'] = questions_df_ls
  df['label'] = labels_df_ls
  df.to_csv(questions_df)

  return intents, df

df = pd.read_csv('data_titles.csv')
intents, questions = generate_intents_and_df(df)