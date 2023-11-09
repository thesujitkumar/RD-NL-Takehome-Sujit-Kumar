#!/usr/bin/env python
# coding: utf-8

# In[211]:


import pandas as pd


# In[212]:


import pandas as pd
import pickle
import sklearn
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
import os
from tqdm import tqdm
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
import argparse
from nltk import word_tokenize, pos_tag
from collections import Counter


# In[ ]:





# In[213]:


nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')


# In[214]:


import re
def striphtml(data):
    p = re.compile(r'<(.*)>.*?|<(.*) />')
    return p.sub('', data)


# # Function for Text cleaning

# In[215]:


def preprocess(sentence):
    #print("setence before parsing",sentence)
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence =  striphtml(sentence)
#     sentence = re.findall(r'http\S+', sentence)
#     sentence = re.sub(r'http\S+', '', sentence, flags=re.MULTILINE)
    sentence=sentence.replace('{html}',"")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 1 ]
#     filtered_words = [w for w in filtered_words if w not in stopwords.words('english')]
    #stem_words=[stemmer.stem(w) for w in filtered_words]
#     lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    #print("filtered word"," ".join(lemma_words))
    return " ".join(filtered_words)


# # Load Dataset

# In[216]:


train=pd.read_csv("train.csv")


# In[217]:


test=pd.read_csv("test.csv")


# In[218]:


dev=pd.read_csv("dev.csv")


# # Check the Basic Distribution of Dtatasets

# In[219]:


len(train)


# In[220]:


train.columns


# In[221]:


train["class"].value_counts()


# In[222]:


len(test)


# In[223]:


test["class"].value_counts()


# In[224]:


len(dev)


# In[225]:


dev["class"].value_counts()


# # Change the type to 0 and 1: Binary
# # Assign  Human generated text to class   0 label
# # Assign  Human generated text to class   1 label

# In[226]:


train.loc[ train['class'] == 'human', 'class'] = 0
train.loc[ train['class'] == 'bot', 'class'] = 1
dev.loc[ dev['class'] == 'human', 'class'] = 0
dev.loc[ dev['class'] == 'bot', 'class'] = 1
test.loc[ test['class'] == 'human', 'class'] = 0
test.loc[ test['class'] == 'bot', 'class'] = 1


# In[227]:


test["class"].value_counts()


# In[228]:


dev["class"].value_counts()


# # Save updated Binary class Datatsets for further computations

# In[229]:


train.to_csv("train_bin.csv", index=False)


# In[230]:


test.to_csv("test_bin.csv", index=False)


# In[231]:


dev.to_csv("dev_bin.csv", index=False)


# # Apply preprocessing over datatsets

# In[232]:


train['text'][0]


# # Preprocessing over training datatsets

# In[233]:


for i in range(len(train)):
    train['text'][i] = preprocess (train['text'][i])


# In[234]:


train['text'][1]


# # Preprocessing over test datatset

# In[235]:


for i in range(len(test)):
    test['text'][i] = preprocess (test['text'][i])


# In[236]:


test['text'][1]


# # # Preprocessing over test datatset

# In[237]:


for i in range(len(dev)):
    dev['text'][i] = preprocess (dev['text'][i])


# In[238]:


dev['text'][1]


# # Save preprocessed  Datatsets for further computations

# In[239]:


train.to_csv("train_bin_cleaned.csv", index=False)


# In[240]:


test.to_csv("test_bin_cleaned.csv", index=False)


# In[241]:


dev.to_csv("dev_bin_cleaned.csv", index=False)


# # Exeract Part of Speech Tagging based feature 

# # Function to exteract pos_tagging feature

# In[242]:


def pos_feature(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Perform POS tagging
    pos_tags = pos_tag(words)

    # Define a set of the 12 major POS tags
    major_pos_tags = {
        'N': 'Noun',
        'V': 'Verb',
        'R': 'Adverb',
        'J': 'Adjective',
        'P': 'Pronoun',
        'D': 'Determiner',
        'C': 'Conjunction',
        'X': 'Preposition',
        'U': 'Interjection',
        'M': 'Number',
        'S': 'Punctuation',
        'F': 'Foreign'
    }

    # Initialize a counter for each major POS tag
    pos_counts = Counter({tag: 0 for tag in major_pos_tags.values()})

    # Count the occurrences of each major POS tag
    for _, tag in pos_tags:
        major_tag = tag[0]
        if major_tag in major_pos_tags:
            pos_counts[major_pos_tags[major_tag]] += 1

    # Print the counts of major POS tags
#     print("Major POS Tag Counts:")
    feature_list=[]
    for tag, count in pos_counts.items():
#         print(f"{tag}: {count}")
        feature_list.append(count)
    return feature_list
        


# # Exetract pos_tagging feature seperately for human generated and bot generated text for comparative study

# In[244]:


train_pos_feature=[]  # exteract pos_tagging features for training datatsets
for i in range(len(train)):
    train_pos_feature.append(pos_feature(train['text'][i]))


# In[245]:


# seperate training and testing features
human = train[train['class'] == 0]
bot = train[train['class'] == 1]


# In[246]:


train_human_gen_text_feature=[] # exteract pos_tagging features for human generated text of training datatsets
for text in human["text"]:
    train_human_gen_text_feature.append(pos_feature(text))


# In[248]:


len(train_human_gen_text_feature)


# In[249]:


train_human_gen_text_feature


# In[250]:


#convert list of list to numpy array for count of human generated text pos tagging feature feature
import numpy as np
u= np.array(train_human_gen_text_feature)


# In[251]:


train_bot_gen_text_feature=[] #exteract pos_tagging features for bot generated text of training datatsets
for text in bot["text"]:
    train_bot_gen_text_feature.append(pos_feature(text))


# In[252]:


#convert list of list to numpy array for count of boat feature
import numpy as np
x= np.array(train_bot_gen_text_feature)


# In[253]:


x


# # compuare the different post tagging count in human generated text and bot generated text

# In[254]:


# count the total number of time pos_tag appeared in boat generated text
# number of nonzero pos count in each row for boat generated text
result= x.sum(axis=0)
result


# In[255]:


len(x)


# In[256]:


# count the total number of time pos_tag appeared in human generated text
# number of nonzero pos count in each row for human generated text
result= u.sum(axis=0)
result


# In[257]:


len(u)


# # count the number of pos tagging  appeared in each bot generated text.

# In[260]:


non_zero_bot=[]


# In[261]:


# number of nonzero pos count in each row for boat generated text
for i in range(len(x)):
    
    k= np.count_nonzero(x[i,:])
    non_zero_bot.append(k)


# In[262]:


len(non_zero_bot)


# In[263]:


df=pd.DataFrame(non_zero_bot)
df.describe()


# # count the number of pos tagging  appeared in each human generated text.

# In[265]:


# number of nonzero pos count in each row for human generated text
non_zero_human=[]
for i in range(len(u)):
    
    q= np.count_nonzero(u[i,:])
    non_zero_human.append(k)


# In[266]:


df1= pd.DataFrame(non_zero_human)


# In[267]:


df1.describe()


# # Function to check the spelling error in a text doccument and return number of words with mistakes spelling

# In[269]:


import enchant

# Create a dictionary object for your preferred language (e.g., English)
# You can specify the language using an appropriate language code.
# For English, 'en_US' is used.
dictionary = enchant.Dict("en_US")

# Sample text with potential spelling errors
# text = "This is an exmaple of a sentence with speling errors."

# Split the text into words
words = text.split()

# Check each word for spelling errors
print(len(train))
train_spelling_error=[]
for i in range(len(train)):
    spelling_errors = []
    
    for word in train['text'][i].split():
        if not dictionary.check(word):
            spelling_errors.append(word)
    train_spelling_error.append([len(spelling_errors)])
    
test_spelling_error=[]
for i in range(len(test)):
    spelling_errors = []
    
    for word in test['text'][i].split():
        if not dictionary.check(word):
            spelling_errors.append(word)
    test_spelling_error.append([len(spelling_errors)])
    

# # Print the list of spelling errors
# print("Spelling Errors:", spelling_errors)


# In[270]:


len(train_spelling_error)


# In[271]:


train_spelling_error


# In[272]:


test_spelling_error


# In[89]:


#get_ipython().system('pip install textstat')


# # Eastimate the different Readability score of text

# In[273]:


from textstat import flesch_kincaid_grade
from textstat import gunning_fog
from textstat import smog_index
from textstat import dale_chall_readability_score

# Sample text
train_readability_features = []
for i in range(len(train)):
    #temp = []
    # Calculate the Flesch-Kincaid Grade Level
    grade_level = flesch_kincaid_grade(train['text'][i])
    grade_level1 = gunning_fog(train['text'][i])
    grade_level2 = smog_index(train['text'][i])
    grade_level2 = smog_index(train['text'][i])
    grade_level3 = dale_chall_readability_score(train['text'][i])
    temp = [grade_level, grade_level1, grade_level2, grade_level3]
    train_readability_features.append(temp)
    
train_readability_features = np.array(train_readability_features)


#print(train_readability_features)
print('------------------------------------------------')
test_readability_features = []
for i in range(len(test)):
    #temp = []
    # Calculate the Flesch-Kincaid Grade Level
    grade_level = flesch_kincaid_grade(test['text'][i])
    grade_level1 = gunning_fog(test['text'][i])
    grade_level2 = smog_index(test['text'][i])
    grade_level2 = smog_index(test['text'][i])
    grade_level3 = dale_chall_readability_score(test['text'][i])
    temp = [grade_level, grade_level1, grade_level2, grade_level3]
    test_readability_features.append(temp)


#print(test_readability_features)
test_readability_features = np.array(test_readability_features)

# print("Flesch-Kincaid Grade Level:", grade_level)
# print("Flesch-Kincaid Grade Level:", grade_level1 )
# print("Flesch-Kincaid Grade Level:", grade_level2 )
# print("Flesch-Kincaid Grade Level:", grade_level3 )


# # Exteract the post tagging features for training datatset

# In[101]:


train_pos_features=[]
for i in range(len(train)):
    train_pos_features.append(pos_feature(train['text'][i]))
train_pos_features = np.array(train_pos_features)


# # Exteract the post tagging features for text datatset

# In[275]:


test_pos_features=[]
for i in range(len(test)):
    test_pos_features.append(pos_feature(test['text'][i]))
test_pos_features = np.array(test_pos_features)


# In[276]:


train_pos_features


# In[277]:


test_pos_features


# # Exteract features based number of different Post_tagg appeared in text .

# In[285]:


train_u= np.array(train_pos_features)
test_v= np.array(test_pos_features)


# In[292]:


train_total_pos_count=[]
for i in range(len(train_u)):
    
    k= np.count_nonzero(train_u[i,:])
    train_total_pos_count.append([k])


# In[293]:


train_total_pos_count


# In[294]:


test_total_pos_count=[]
for i in range(len(test_v)):
    
    k= np.count_nonzero(test_v[i,:])
    test_total_pos_count.append([k])


# In[295]:


test_total_pos_count


# # Concantenate all the features and form a feature vector fot training and testing

# In[303]:


train_features = np.concatenate((train_readability_features, train_pos_features,train_total_pos_count,train_spelling_error), axis=1)


# In[304]:


train_features.shape


# In[305]:


test_features = np.concatenate((test_readability_features, test_pos_features,test_total_pos_count, test_spelling_error), axis=1)


# In[306]:


test_features.shape


# # Dump all the exteracted feature over training and text datatsets for future experiments

# In[309]:


df= pd.DataFrame(train_features)
df.to_csv("train_features.csv",index=None)


# In[310]:


df1= pd.DataFrame(test_features)
df1.to_csv("test_features.csv",index=None)


# In[313]:


y_train= np.array(train["class"].tolist())


# In[314]:


y_train


# In[315]:


y_test= np.array(test["class"].tolist())


# In[316]:


y_test


# # Train a Support vector machine (SVM) classifier with exteracted features

# In[318]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC



# # Model Evaluation fucntion with accuracy, f-measure and class wise f1 score

# In[325]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def my_score(y_test, y_pred):
    acc =accuracy_score(y_test, y_pred)
    f1= f1_score(y_test, y_pred, average='macro')
    
    f1_classwise = f1_score(y_test, y_pred, average=None)
    
    
    return acc,f1,f1_classwise


# # Fit SVM

# In[323]:


clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(train_features, y_train)
y_pred = clf.predict(test_features)


# In[327]:

print("performance of SVM  classifier")
score= my_score(y_test, y_pred)
print(score)




# # Fit Ensemble Model Adaboost classifier with exteracted features

# In[348]:


from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=140, random_state=0)


# In[349]:


clf.fit(train_features, y_train)


# In[350]:


y_pred = clf.predict(test_features)


# In[351]:

print("performance of Adaboost  classifier")
score= my_score(y_test, y_pred)
print(score)


# # verify confucion matrix to understand predictions

# In[352]:


from sklearn.metrics import confusion_matrix


# In[353]:


x=confusion_matrix(y_test, y_pred)
print("confusion matrix", x)

# # Fit Ensemble Model Gradient Boosting  classifier with exteracted features

# In[354]:


from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier


# In[355]:


clf = GradientBoostingClassifier(n_estimators=150, learning_rate=1.0,
    max_depth=1, random_state=0).fit(train_features, y_train)


# In[356]:


y_pred = clf.predict(test_features)


# In[357]:

print("performance of Gradient Boosting  classifier")
score= my_score(y_test, y_pred)
print(score)


# In[358]:

print("performance of Gradient Boosting  classifier")
x= confusion_matrix(y_test, y_pred)
print("confusion matrix", x)


# In[ ]:




