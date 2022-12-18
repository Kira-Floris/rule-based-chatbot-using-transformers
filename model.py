# setup
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# import
# import libraries
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
import json
from transformers import RobertaTokenizer, RobertaModel, DistilBertTokenizer, DistilBertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import lr_scheduler
import random
from sklearn.preprocessing import LabelEncoder

questions = pd.read_csv('questions.csv')

device = torch.device('cuda')

le = LabelEncoder()
questions['label'] = le.fit_transform(questions['label'])

train_text, train_labels = questions['text'], questions['label']

# calculating number of unique labels
unique_labels = len(set(questions['label']))

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

max_seq_len = 16

# tokenize and encode sequences in training set
tokens_train = tokenizer(
    train_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length = True,
    truncation = True,
    return_token_type_ids = False
)

# train data to tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# dataloaders
batch_size = 20
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# model

class BERT_Arch(nn.Module):
  def __init__(self, bert, dropout=0.2, hl1=768, hl2=512, hl3=512, out=3):
    super(BERT_Arch, self).__init__()
    self.bert = bert
    self.dropout = nn.Dropout(dropout)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(hl1, hl2)
    self.fc2 = nn.Linear(hl2, hl3)
    self.fc3 = nn.Linear(hl3, out)
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

# freezing parameters
for param in bert.parameters():
  param.requires_grad = False

model = BERT_Arch(bert, out=unique_labels)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr = 1e-3)

class_wts = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)

# convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)
# loss function
cross_entropy = nn.NLLLoss(weight=weights) 

train_losses=[]
# number of training epochs
epochs = 400
# We can also use learning rate scheduler to achieve better results
lr_sch = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# function to train the model
def train():
  
  model.train()
  total_loss = 0
  
  # empty list to save model predictions
  total_preds=[]
  
  # iterate over batches
  for step,batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('\tBatch {:>5,} out of {:>5,}.'.format(step,len(train_dataloader)))
      print(f'\tloss: {total_loss/len(train_dataloader):.3f}')
    # push the batch to gpu
    batch = [r.to(device) for r in batch] 
    sent_id, mask, labels = batch
    # get model predictions for the current batch
    preds = model(sent_id, mask)
    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)
    # add on to the total loss
    total_loss = total_loss + loss.item()
    # backward pass to calculate the gradients
    loss.backward()
    # clip the the gradients to 1.0. It helps in preventing the    exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # update parameters
    optimizer.step()
    # clear calculated gradients
    optimizer.zero_grad()
  
    # We are not using learning rate scheduler as of now
    # lr_sch.step()
    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()
    # append the model predictions
    total_preds.append(preds)
  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
    
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)
  #returns the loss and predictions
  return avg_loss, total_preds

for epoch in range(epochs):
     
    print('Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ = train()
    
    # append training and validation loss
    train_losses.append(train_loss)
    # it can make your experiment reproducible, similar to set  random seed to all options where there needs a random seed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f'\nTraining Loss: {train_loss:.3f}')

model_path = 'model.pth'
torch.save(model, model_path)

import random

def get_prediction(str):
 str = re.sub(r'[^a-zA-Z ]+', '', str)
 test_text = [str]
 model.eval()
 
 tokens_test_data = tokenizer(
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
   preds = model(test_seq.to(device), test_mask.to(device))
 preds = preds.detach().cpu().numpy()
 preds = np.argmax(preds, axis = 1)
 print('Intent Identified: ', le.inverse_transform(preds)[0])
 return le.inverse_transform(preds)[0]
 
def get_response(message): 
  intent = get_prediction(message)
  for i in data['intents']: 
    if i["tag"] == intent:
      result = i["responses"]
      break
  print(f"Response : {result}")
  return "Intent: "+ intent + '\n' + "Response: " + result

get_response('what is rura')