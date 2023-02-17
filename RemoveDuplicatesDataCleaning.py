#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary libraries

# In[1]:


import re
import os
import time
import spacy
import string
import warnings
import itertools
import numpy as np
import contractions
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from IPython.display import display # just to display


# In[2]:


start = time.time()


# In[3]:


warnings.filterwarnings('ignore')


# ## Importing the DataFrame 

# In[4]:


df = pd.read_csv("./news.csv")
df.head()


# ## Analyzing the DataFrame 

# In[5]:


df.isna().sum()


# In[6]:


df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)


# In[7]:


df.isna().sum()


# In[8]:


df.drop(['title', 'published_at', 'source', 'topic'], axis=1, inplace=True)


# In[9]:


df.head()


# ## Performing the DataCleaning

# In[10]:


small_model = spacy.load('en_core_web_sm')
medium_model = spacy.load('en_core_web_md')
large_model = spacy.load('en_core_web_lg')


# In[11]:


class DataCleaning:
    def __init__(self) -> None:
        self.__stopwords = stopwords.words("english")
        self.__punctuations = string.punctuation.replace(".", "")
    
    
    def __remove_html_tags_entities(self, data: str) -> str:
        """
        to remove HTML tags, entities and links if any from the data
        :param data: string with HTML tags, entities and links
        :return: string without HTML tags, entities and links
        """
        data = BeautifulSoup(str(data), "lxml").get_text(strip=True)
        data = data.encode("ascii", "ignore").decode()
        return data.strip()
    
    
    def __remove_last_fullstop(self, data: str) -> str:
        """
        to remove the last fullstop if present, because in later stages 
        while splitting the data on the basis of fullstop, the last 
        fullstop leads to generation of empty string
        :param data: string which may have fullstop at end
        :return: string without fullstop at the end
        """
        data = self.__remove_html_tags_entities(data)
        if data[-1] == '.':
            data = data[:-1]
            return data
        else:
            return data
        
        
    def __remove_links(self, data: str) -> str: 
        """
        to remove any website links that might get scrapped with text
        :param data: string with or without links
        :return: string without links
        """
        data = self.__remove_last_fullstop(data)
        return re.sub(r"(http(s)?:\/\/.)+(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}(\.)*[ a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&\/=]*)", '', data)
    
    
    def __general_preprocessing(self, data: str) -> str:
        """
        to remove any \n, \t, ]r characters
        :param data: string with any irrelevant characters
        :return: string without any irrelevant characters
        """
        data = self.__remove_links(data)
        data = re.sub(r"(\n)+", ".", data)
        data = re.sub("[\t\r]", ' ', data)
        data = re.sub("(  )+", " ", data)
        return data
    
    
    def __retain_fullstops_within_quotes(self, data: str) -> str:
        """
        to retain the fullstops without the double quotes
        :param data: string with fullstops inside double quotes
        :return: string in which fullstops inside double quotes are replaced with ''
        """
        data = self.__general_preprocessing(data)
        return re.sub(r'"[^"]*"', lambda m: m.group(0).replace('.', '') + ".", data)

    
    
    def __remove_contractions(self, data: str) -> str:
        """
        to convert contractions to expanded words
        :param data: string with contractions, ex: "It's"
        :return: string without contractions, ex: "It's" -> "It is"
        """
        data = self.__retain_fullstops_within_quotes(data)
        return " ".join([contractions.fix(i) for i in data.split(" ")])
    
    
    def __remove_empty_sentences(self, data: str):
        """
        to remove any sentence that has length zero or it's just more than 1 consecutive fullstops
        :param data: string with or without consecutive fullstops
        :return: string without empty sentence
        """
        data = self.__remove_contractions(data)
        data = re.sub(" \.", ".", data)
        data = re.sub("\. ", ".", data)
        data = re.sub(" \. ", ".", data)
        data = re.sub("(\.)+", ".", data)
        return ".".join(sentence for sentence in data.split(".") if len(sentence) != 0)
   

    def __remove_punctuations(self, data: str) -> str:
        """
        to remove punctuations
        Note: Here, we are neither inserting nor removing any fullstop. Therefore no. of sentences 
        in clean data as well as structured data remains same after applying the pre-processing.
        :param data: string which has punctuations
        :return: string without punctuations
        """
        data = data.lower()
        return data.translate(data.maketrans('', '', self.__punctuations))
    
    
    def __remove_stopwords(self, data: str) -> str:
        """
        to remove stopwords from the sentence
        :param data: string with stopwords, e.g.: 'This is great'
        :return: string without stopwords, e.g.: 'This great'
        """
        data = self.__remove_punctuations(data)
        return " ".join([i for i in data.split(" ") if i not in self.__stopwords])
        
        
    def clean_data_nf(self, data: str) -> str:
        """
        to perform the appropriate preprocessing on the data
        :param data: raw data
        :return: clean data which can be used for further processing
        """
        data = self.__remove_empty_sentences(data)
        return data
    
    def structured_data_nf(self, data: str) -> str:
        """
        to remove the punctuations and stopwords from the cleaned data
        :param data: cleaned data
        :return: structured data
        """
        data = self.__remove_stopwords(data)
        return data
    
    
    def __sentence_formatting(self, data: list[str]) -> str:
        flag, sentences = 0, []
        for i in range(len(data)):
            if flag == 1:
                flag = 0
                continue
            sentence = data[i]
            total_words = len(sentence.split(" "))
            if total_words > 25 and sentence not in sentences:
                sentences.append(sentence)
            elif 0 < total_words <= 25 and sentence not in sentences:
                if i == 0:
                    sentences.append(data[i] + " " + data[i+1])
                elif i == len(data)-1:
                    sentences.append(data[i-1] + " " + data[i])
                elif len(data[i-1].split(" ")) <= len(data[i+1].split(" ")):
                    sentences.pop(-1)
                    sentences.append(data[i-1] + " " + data[i])
                elif len(data[i-1].split(" ")) > len(data[i+1].split(" ")):
                    sentences.append(data[i] + " " + data[i+1])
                    flag = 1
        return '.'.join(sentences)
    

    
    def __sentence_boundary_detection_small_model(self, data: str) -> str:
        doc = small_model(data)
        sentences = list(doc.sents)
        data = ""
        for i in sentences:
            data += str(i).strip() + "."
        data = data.split(".")
        sentences = self.__sentence_formatting(data)
        data = []
        for i in sentences.split("."):
            if len(i) != 0:
                data.append(i)
        sentences = ".".join(data)
        sentences = re.sub("(  )+", " ", sentences)
        savg = len(sentences.split(" "))/len(sentences.split("."))
        return sentences.strip(), savg
    
    def __sentence_boundary_detection_medium_model(self, data: str) -> str:
        doc = medium_model(data)
        sentences = list(doc.sents)
        data = ""
        for i in sentences:
            data += str(i).strip() + "."
        data = data.split(".")
        sentences = self.__sentence_formatting(data)
        data = []
        for i in sentences.split("."):
            if len(i) != 0:
                data.append(i)
        sentences = ".".join(data)
        sentences = re.sub("(  )+", " ", sentences)
        mavg = len(sentences.split(" "))/len(sentences.split("."))
        return sentences.strip(), mavg
    
    
    def __sentence_boundary_detection_large_model(self, data: str) -> str:
        doc = large_model(data)
        sentences = list(doc.sents)
        data = ""
        for i in sentences:
            data += str(i).strip() + "."
        data = data.split(".")
        sentences = self.__sentence_formatting(data)
        data = []
        for i in sentences.split("."):
            if len(i) != 0:
                data.append(i)
        sentences = ".".join(data)
        sentences = re.sub("(  )+", " ", sentences)
        lavg = len(sentences.split(" "))/len(sentences.split("."))
        return sentences.strip(), lavg
        
    
    def __use_appropriate_model(self, data: str) -> str:
        small, savg = self.__sentence_boundary_detection_small_model(data)
        # print(f"SMALL:\n{small}")
        medium, mavg = self.__sentence_boundary_detection_medium_model(data)
        # print(f"MEDIUM:\n{medium}")
        large, lavg = self.__sentence_boundary_detection_large_model(data)
        # print(f"LARGE:\n{large}")
        selection_criteia = [[savg, small], [mavg, medium], [lavg, large]]
        selection_criteia.sort()
        final_selected_sentence = self.__sentence_formatting(selection_criteia[0][1].split("."))
        # print(final_selected_sentence)
        return final_selected_sentence
    
    
    def sentence_boundary_detection(self) -> None:
        for no_of_sentence in range(1, 20):
            indices = df[df['no_of_sentence'] == no_of_sentence].index
            for i in indices:
                data = df['clean_data'][i]
                res = len(data.split("."))
                print(f"Index:{i} Before Cleaning: {res}")
                data = self.__use_appropriate_model(data)
                df['clean_data'][i] = data
                res, res1 = len(data.split(".")), len(df['clean_data'][i].split("."))
                print(f"After Cleaning {res} and {res1}")
                df['structured_data'][i] = self.structured_data_nf(df['clean_data'][i])
                print("-"*60)
            df['no_of_sentence'] = df['structured_data'].apply(lambda x: len(x.split(".")))
        
    def remove_unnecessary_fullstops(self) -> None:
        max_sentences_in_a_row = df.sort_values(by='no_of_sentence').max()[2]+1
        for no_of_sentence in range(max_sentences_in_a_row, 9, -1):
            indices = df[df['no_of_sentence'] == no_of_sentence].index
            for i in indices:
                data = df['clean_data'][i]
                res = len(data.split("."))
                print(f"Index:{i} Before Cleaning: {res}")
                data = self.__sentence_formatting(data.split("."))
                df['clean_data'][i] = data
                res, res1 = len(data.split(".")), len(df['clean_data'][i].split("."))
                print(f"After Cleaning {res} and {res1}")
                df['structured_data'][i] = self.structured_data_nf(df['clean_data'][i])
                print("-"*60)
            df['no_of_sentence'] = df['structured_data'].apply(lambda x: len(x.split(".")))
    
    
    def clean_data_f(self, data: str) -> str:
        return self.__sentence_formatting(data.split("."))


# In[12]:


dc = DataCleaning()


# In[13]:


clean_data = dc.clean_data_nf(df['content'][0])
clean_data.split(".")


# In[14]:


df['clean_data'] = df['content'].apply(dc.clean_data_nf)
df.head()


# In[15]:


structured_data = dc.structured_data_nf(clean_data)
structured_data.split(".")


# In[16]:


df['structured_data'] = df['clean_data'].apply(dc.structured_data_nf)
df.head()


# In[17]:


df.drop(['content'], axis=1, inplace=True)


# In[18]:


df.head()


# In[19]:


df['no_of_sentence'] = df['structured_data'].apply(lambda x: len(x.split(".")))
df.head()


# In[20]:


df.sort_values(by='no_of_sentence')


# In[21]:


df['clean_data'][150].split(".")


# In[22]:


df[df['no_of_sentence'] == 1].index


# In[23]:


get_ipython().run_cell_magic('time', '', 'dc.sentence_boundary_detection()')


# In[24]:


df['clean_data'][367].split(".")


# In[25]:


print(df.sort_values(by='no_of_sentence').max()[2])


# In[26]:


df['clean_data'][534].split(".")


# In[27]:


get_ipython().run_cell_magic('time', '', 'dc.remove_unnecessary_fullstops()')


# In[28]:


df['clean_data'][534].split(".")


# In[29]:


df.sort_values(by='no_of_sentence')


# In[30]:


df['clean_data'][594].split(".")


# In[31]:


len(df['clean_data'][533].split("."))


# In[32]:


len(df['structured_data'][533].split("."))


# In[33]:


df.head()


# In[34]:


df.shape


# In[35]:


df.isna().sum()


# In[36]:


df.shape


# In[37]:


df.sort_values(by='no_of_sentence')


# In[38]:


df.sort_values(by='no_of_sentence', ascending=False)


# In[39]:


df['clean_data'][529]


# In[40]:


df['clean_data'] = df['clean_data'].apply(dc.clean_data_f)


# In[41]:


df['structured_data'] = df['clean_data'].apply(dc.structured_data_nf)


# In[42]:


df.sort_values(by='no_of_sentence', ascending=False)


# In[43]:


df.drop(['no_of_sentence'], axis=1, inplace=True)


# In[44]:


df.to_csv('./clean_and_structured_news.csv', index=False)


# In[45]:


print(f"Total time taken in complete program execution: {(int)((time.time()-start)//60)} mins {(int)((time.time()-start)%60)} secs")

