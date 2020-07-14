import pandas as pd
import functools

df1 = pd.DataFrame({'FI1': [2,3,3,4]})
df2 = pd.DataFrame({'FI2': [10,30,3,42]})
df3 = pd.DataFrame({'FI3': [5,9,3,19]})
df4 = pd.DataFrame({'FI4': [8,91,34,29]})

#Combine dfs in a list
dfList = [df1, df2, df3, df4]
#Print each dfs
for i in range(len(dfList)):
    print(dfList[i])

#Merge list of dfs into a single columnwise df using reduce funtion
#reduce() fn here takes merge(fn) and list(iterable) as arguments
#merge being applied to first two dfs in a list firstly, then takes subsequent list one at a time sequentially
#E.g: (((1 merge 2) merge 3) merge 4), gives the final output
dflist_comb=functools.reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), dfList)

#Function to reduce memory usage based on datatypes
import numpy as np

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
##Preprocessing a Text document
#Concatenate lines from row/cell which are seperated by linebreaks into a single line. This helps in defining what is sentence in an NLP tasks
#Remove duplicate lines
#Remove leading and Trailing spaces in a text(strip()), multiple spaces
#Remove blank lines

import numpy as np
import pandas as pd
import re
from functools import reduce

textlst = ['Natural   language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data \n Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.',
       'The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods \n In 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence \n The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English.   The authors claimed that within three or five years, machine    translation would be a solved problem.  ', ' ', ' ','Natural   language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data \n Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.', 'Damit Natural Language Processing funktioniert, muss zunächst an der Spracherkennung gearbeitet werden. NLP wird als zukunftsträchtige Technologie im Bereich HCI für die Steuerung von Geräten oder Webanwendungen gesehen. So basierte zum Beispiel die Arbeit von Chatbots oder digitalen Sprachassistenten auf diesem Prinzip.']
df = pd.DataFrame(textlst,columns =['Text'])
join_lines = []
for i in df['Text']:
    clean=i.replace("\n",". ")
    clean=clean.strip()
    clean = re.sub(' +', ' ',clean)
    join_lines.append(clean)
join_lines = list(filter(None, join_lines))
df_clean = pd.DataFrame(join_lines,columns =['Text'])
df_clean.drop_duplicates(subset ="Text",
                     keep = 'first', inplace = True)

#Remove accentuation from the given string.
#Helps in analyzing corpus which is multi-linguistic
import unicodedata
import sys
if sys.version_info[0] >= 3:
    unicode = str

def deaccent(text):
    """
    Remove accentuation from the given string. Input text is either a unicode string or utf8 encoded bytestring.
    Return input string with accents removed, as unicode.
    >>> deaccent("Šéf chomutovských komunistů dostal poštou bílý prášek")
    u'Sef chomutovskych komunistu dostal postou bily prasek'
    """
    if not isinstance(text, unicode):
        # assume utf8 for byte strings, use default (strict) error handling
        text = text.decode('utf8')
    norm = unicodedata.normalize("NFD", text)
    result = ''.join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)

#deaccent of a string
text = 'Šéf chomutovských komunistů dostal poštou bílý prášek'
deaccent(text)

#deaccent of a list/corpus with n documents
deaccented_sentences = []
for line in join_lines:
    line_lowercase = line.lower()
    accent = deaccent(line_lowercase)
    accent= accent.replace('"','')
    accent=accent.strip()
    accent = re.sub(' +', ' ',accent)
    deaccented_sentences.append(accent)

##
import numpy as np
import pandas as pd
import re
from functools import reduce
#using news_data, publically available
df = pd.read_csv("news_data.csv", encoding = "ISO-8859-1", sep =",")

##Preprocessing a Text document
join_lines = []
for i in df['Text']:
    clean=i.replace("\n",". ")
    clean=clean.strip()
    clean = re.sub(' +', ' ',clean)
    join_lines.append(clean)
join_lines = list(filter(None, join_lines))
df_clean = pd.DataFrame(join_lines,columns =['Text'])
df_clean.drop_duplicates(subset ="Text",
                     keep = 'first', inplace = True)
doc = df_clean['Text']

#Extract sentences from each document by split lines using RE
#Get the number of sentences in each document
sentences = []
doc_len = []
i = 0
num_docs = 0
for line in doc:
    sent_split = re.split(r"[.:;!?]",line)
    sent_split = list(filter(None, sent_split))
    sent_strip = []
    for sent in sent_split:
        sent= sent.replace('"','')
        sent=sent.strip()
        sent_strip.append(re.sub(' +', ' ',sent))
    sent_strip = list(filter(None, sent_strip))
    sent_split = sent_strip
    i = len(sent_split)
    doc_len.append(i)
    sentences.extend(sent_split)
    num_docs += 1
df_clean_len = pd.DataFrame({'Text': doc,'Text_len': doc_len})

#get stop-words
f = open("stopwords.txt")
stopwords = set()
for line in f:
    stopwords.add(line.rstrip())
#print(stopwords)

# remove stop-words
doc_stop_rem = []
for line in doc:
    doc_stop_rem.append(' '.join([word for word in line.split() if word not in stopwords]))

df_clean_len = pd.DataFrame({'Text': doc_stop_rem})

#convert into lowercase and remove punctuation from dataframe
#https://stackoverflow.com/questions/47947438/preprocessing-string-data-in-pandas-dataframe
df_clean_len.loc[:,"Text"] = df_clean_len.Text.apply(lambda x : str.lower(x))
import re
df_clean_len.loc[:,"Text"] = df_clean_len.Text.apply(lambda x : " ".join(re.findall('[\w]+',x)))

clean = df_clean_len['Text']

#Get the total count of words in the corpus
#Get the total count of words in each document
#Counter to get the count of each word
#count of unique_words in the corpus

from collections import Counter
no_words_corpus = 0
no_words_doc = []
word_count = Counter()
for doc_index, doc in enumerate(clean):
#    no_words_doc = []
    words = doc.split()
    no_words = len(words)
    no_words_doc.append(no_words)
    for word_index, word in enumerate(words):
        word_count[word] += 1
        no_words_corpus += 1
unique_words = len(word_count)

#get the word_index for each document
doc_word_indices = []
for doc_index, doc in enumerate(clean):
    words = doc.split()
    word_indices = []
    for word_index, word in enumerate(words):
        word_indices.append(word_index)
    doc_word_indices.append(word_indices)

###
#get the words and phrases(len 2/3 of contiguous word)s appearing greater than min_frequency
##Identify phrases as mentioned in El-Kishky, Ahmed, et al. "Scalable topical phrase mining from text corpora."
##Reference code taken from https://github.com/anirudyd/topmine

freq_counter = word_count
n = 2
min_freq=4
max_phrase_len=3
#indicates phrases of length 2

while(len(clean) > 0):
    temp_documents = []
    new_doc_word_indices = []
    for d_i,doc in enumerate(clean):
        new_word_indices = []
        word_indices = doc_word_indices[d_i]
        for index in word_indices:
            words = doc.split()
            if index+n-2 < len(words):
                key = ""
                for i in range(index, index+n-2+1):
                    if i == index+n-2:
                        key = key + words[i]
                    else:
                        key = key + words[i] + " "

                if freq_counter[key] >= min_freq:
                    new_word_indices.append(index)

        new_doc_word_indices.append(new_word_indices)
        temp_documents.append(doc)
        words = doc.split()
        for idx, i in enumerate(new_word_indices[:-1]):
            phrase = ""
            if (new_word_indices[idx+1] == i + 1):
                for idx in range(i, i+n):
                    if idx == i+n-1:
                        phrase += words[idx]
                    else:
                        phrase += words[idx] + " "
            freq_counter[phrase] += 1

    documents = temp_documents
    n += 1
    if n == max_phrase_len:
        break

freq_counter = Counter(x for x in freq_counter.elements() if freq_counter[x] >= min_freq)

##
#filter only phrases(len 2/3 of contiguous word)s appearing greater than min_frequency in order of count
frequent_phrases = []
for key,value in freq_counter.most_common():
    if value >= min_freq and len(key.split(" "))>1:
        frequent_phrases.append((key, value))
    elif value < min_freq:
        break

##get documentwise list of all applicable phrases with its count
df_clean_p = pd.DataFrame({'Text': documents})
clean_p = df_clean_p['Text']
doc_phrases = {}
phrase_freq_doc_df = pd.DataFrame()

for docid, doc in enumerate(clean_p):
    phrases = doc.split()
    s = pd.DataFrame()
    for index, word in enumerate(phrases[:-1]):
        doc_phrases = {}
        phrase = phrases[index]+" "+phrases[index+1]
        doc_phrases[phrase] = freq_counter[phrase]
        s = pd.Series(doc_phrases, name='freq')
        s.index.name = 'phrases'
        s=s.reset_index()
        s['doc_id'] = docid
        phrase_freq_doc_df = phrase_freq_doc_df.append(s)
phrase_freq_doc_df = phrase_freq_doc_df[phrase_freq_doc_df['freq'] > 0]