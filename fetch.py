from bs4 import BeautifulSoup as bs
import requests
import re

import sys
sys.setrecursionlimit(1000000000)

import warnings

warnings.filterwarnings("ignore")


def get_full_link(main, link):
  full = re.search('^http', link)
  link = str(link)
  if full !=None:
    return link
  else:
    if main:
      main = str(main)
      if list(str(link))[0]=='i': return main+'/'+link
      return main+link
    else: return link
    
def remove_urls(link):
  remove = ['facebook', 'youtube', 'mail', 'linkedin', 'flickr', 'facebook', 'google']
  for item in remove:
    if re.search(item, str(link)):
      return False
  return True

def get_links(link, company, all_links=set()):
  try:
    data = requests.get(link, verify=False, timeout=10)
    result = bs(data.text, 'html.parser')
    all_links.add(link)
    [all_links.add(get_full_link(company, a['href'])) for a in result.find_all('a') if remove_urls(a)]
  except Exception as err:
    pass
  return all_links


all_links = get_links('https://rura.rw/index.php?id=23', 'https://rura.rw')

all_links = get_links('https://www.rra.gov.rw/en/home', 'https://www.rra.gov.rw/en', all_links)

all_links = get_links('https://www.rssb.rw/', 'https://www.rssb.rw', all_links)

all_links = get_links('https://www.rema.gov.rw/home', 'https://www.rema.gov.rw', all_links)

all_links = get_links('https://www.rha.gov.rw/', 'https://www.rha.gov.rw', all_links)

all_links = get_links('https://www.gov.rw/', 'https://www.gov.rw', all_links)        

print(f'Finished getting main links, length: {len(all_links)}')

# get more links from the website
for link in all_links:
  all_links = list(get_links(link, None, set(all_links)))
  
# removing urls without http or rw
mixed_urls = all_links
all_links = [item for item in all_links if 'http' and '.rw' in item]
len(all_links)

# creating a temp json for all urls
import json

# saving json
temporary_json = {
    'all_urls':mixed_urls,
    'clean_urls':all_links
}

json_object = json.dumps(temporary_json)
with open('urls.json','w') as f:
  f.write(json_object)
  
print(f'Finished getting inside links, length: {len(all_links)} and saved at urls.json')
# scraping the data
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import re

def check_http(link): return True if('http' in link) else False

def add_new_row(data, df):
  temp = pd.DataFrame(data)
  check = temp.Link.isin(df.Link).astype(int)
  # check if row exists
  if check[0]==1:
    return df
  # if not append
  else:
    df = df.append(temp, ignore_index=True)
    return df

def scrap(link, df=None):
  # check if link has http
  if check_http(link):
    # get link data
    try:
      url_text = requests.get(link, verify=False, timeout=10)
      data = bs(url_text.text, 'html.parser')
      
      # get title
      try:
        title = data.title
      except:
        title = ''

      # remove all style, header, nav and script
      for s in data(['script','style','header','link','nav','meta','head']):
        s.decompose()

      # get main body
      body = data.find_all('div')
      
      # splitting body by \n
      body_extended = []
      [body_extended.extend(item.text.split('\n')) for item in body]
      body_extended = [item for item in body_extended if item!='']

      # match nber of titles and links to size of body
      body_length = len(body_extended)
      title_extended = [title] * body_length
      link_extended = [link] * body_length

      # creating a dataframe and appending to df
      temp = {
          'Title':title_extended, 
          'Link':link_extended, 
          'Document':body_extended
          }
      df = add_new_row(temp, df)
      return df
    except Exception as err:
      return df
  else: 
    return df
  

# initializing the data
format = {
    'Title':[], 
    'Link':[], 
    'Document':[]
    }
df = pd.DataFrame(data=format)
df.head()

print('starting scraping...')

# fetching data from the urls
temp_list = []
for link in all_links:
  temp_list.append(scrap(link, df=df))

df = df.append(temp_list)
df.to_csv('data.csv')

print(f'Finished scraping with length: {len(df)} and saved at data.csv')