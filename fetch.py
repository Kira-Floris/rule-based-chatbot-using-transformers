import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import re

import warnings
warnings.filterwarnings('ignore')

def verify_full_link(link, main, main_link=None):
  if link=='#':
    return main_link
  if link[-2]=='.':
    link = link[:-2]+link[-1]
  if re.search('http', str(link)):
    return str(link)
  else:
    if link[0]=='?' or '/':
      return main[:-1]+link
    else:
      return main+link

def verify_title(title):
  if len(str(title))>0: return str(title)
  else: return None

# cleaning titles
def preprocessing(titles : list):
  cleaned_titles = []
  for title in titles:
    title = title.lower()
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    html_pattern = re.compile('<.*?>')
    title = url_pattern.sub(r'', title)
    title = html_pattern.sub(r'', title)
    title = re.sub(r"[^\w\d'\s]+", ' ', title)
    cleaned_titles.append(title)
  return cleaned_titles

def get_links(link, main=None):
  content = requests.get(link, verify=False, timeout=10).content
  page = bs(content, 'html.parser')
  links = page.find_all('a')
  titles = [verify_title(item.text) for item in links]
  links_href = [verify_full_link(item['href'],main, link) for item in links]
  # descriptions = [get_description(item) for item in links_href]
  
  # create dataframe
  df = pd.DataFrame()
  df['title'] = titles
  df['links'] = links_href
  # df['description'] = descriptions

  # drop empty titles
  df = df.dropna(subset=['title']).reset_index(drop=True)

  # drop duplicate 
  df = df.drop_duplicates().reset_index(drop=True)
  df['title'] = preprocessing(df['title'])
  return df

df1 = get_links('https://rura.rw/index.php','https://rura.rw/')
df2 = get_links('https://www.rema.gov.rw/home','https://rema.gov.rw/')

frames = [df1, df2]
# combine dataframes
df = pd.concat(
    frames, 
    ignore_index=True
    )

# save csv file
df.to_csv('data_titles.csv')