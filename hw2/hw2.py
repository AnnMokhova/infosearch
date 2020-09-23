import pandas as pd
import pymorphy2
from nltk.corpus import stopwords
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import Any
from math import log

file = 'answers_base.xlsx'
xl = pd.ExcelFile(file)
df = xl.parse(xl.sheet_names[0])


def preproccesing(text):
    morph = pymorphy2.MorphAnalyzer()
    # проверяем наличие кириллических символов в слове
    clean_text = ' '.join([word for word in str(text).split() if re.search(r"[а-я]", word)])
    clean_text = ' '.join([word for word in clean_text.lower().split()])
    # убираем знаки препинания
    clean_text = clean_text.translate(str.maketrans(dict.fromkeys(string.punctuation))).split()
    clean_text = [re.sub(r'[«|»|…|\ufeff]', r'', word) for word in clean_text]
    clean_text = [re.sub(r'\n', r' ', word) for word in clean_text]
    # лемматизируем
    clean_text = ' '.join([morph.parse(word)[0].normal_form for word in clean_text])
    # убираем стоп-слова
    clean_text = ' '.join([word for word in clean_text.split() if word not in stopwords.words('russian')])
    return clean_text


corpus = [preproccesing(query) for query in df['Текст вопросов']]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()


# функция перевода входяшего запроса в вектор по метрике TF-IDF
def vector_tfidf(query: str, vectorizer: Any) -> Any:
    new_query = []
    for word in query.split():
        if word in vectorizer.vocabulary_.keys():
            new_query.append(word)
    return vectorizer.transform(new_query).toarray()


vector_query = vector_tfidf(preproccesing('как вылечиться от ковида'), vectorizer)
res_tfidf = np.dot(X, vector_query[0]).tolist()


def ranging(res):
    d = {}
    # создаем словарь, где ключ - индекс документа, значение - близость документа с запросом
    index = 0
    for doc in res:
        d[index] = doc
        index += 1
    sorted_by_value = sorted(d.items(), key=lambda x: x[1], reverse=True)

    return sorted_by_value


print('Ранжирование докуменов по близости к запросу по убыванию для tfidf ', ranging(res_tfidf))


def bm25(word, doc, corpus) -> float:
    N = len(corpus)
    n = 0
    for doc in corpus:
        if word in doc.split():
            n += 1
    idf = log((N - n + 0.5) / (n + 0.5))
    k = 2.0
    b = 0.75
    tf = doc.count(word) / len(doc.split())
    avgdl = sum([len(text.split()) for text in corpus]) / len(corpus)
    score = idf * tf * (k + 1) / (tf + k * (1 - b + b * len(doc.split()) / avgdl))
    return score


list_of_words = vectorizer.get_feature_names()
matrix = []
for i in range(len(corpus)):
    a = []
    for j in range(len(list_of_words)):
        w = list_of_words[j]
        d = corpus[i]
        a.append(bm25(w, d, corpus))
    matrix.append(np.array(a))
matrix = np.array(matrix)


# функция перевода входяшего запроса в вектор по метрике BM25
def vector_bm25(query, list_of_words):
    words = list(set(query.split()))
    vector = []
    for i in list_of_words:
        if i in words:
            vector.append(1)
        else:
            vector.append(0)
    return np.array(vector)


vector = vector_bm25(preproccesing('существует ли вакцина'), list_of_words)

res_bm25 = matrix.dot(vector).tolist()

print('Ранжирование докуменов по близости к запросу по убыванию для bm25 ', ranging(res_bm25))
