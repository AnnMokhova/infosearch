import os
import pymorphy2
from nltk.corpus import stopwords
import string
import re
import json
from sklearn.feature_extraction.text import CountVectorizer
import pickle as pkl

curr_dir = os.getcwd()
filepath = os.path.join(curr_dir, 'friends-data')
filepaths = []

for root, dirs, files in os.walk(filepath):
    for name in files:
        filepaths.append(os.path.join(root, name))

num_episodes = len(filepaths)


def preproccesing(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
        morph = pymorphy2.MorphAnalyzer()
        clean_text = ' '.join([word for word in text.lower().split()])
        # убираем знаки препинания
        clean_text = clean_text.translate(str.maketrans(dict.fromkeys(string.punctuation))).split()
        clean_text = [re.sub(r'[«|»|…|\ufeff]', r'', word) for word in clean_text]
        # лемматизируем
        clean_text = ' '.join([morph.parse(word)[0].normal_form for word in clean_text])
        # убираем стоп-слова
        clean_text = ' '.join([word for word in clean_text.split() if word not in stopwords.words('russian')])
        # проверяем наличие кириллических символов в слове
        clean_text = ' '.join([word for word in clean_text.split() if re.search(r"[а-я]", word)])
        return clean_text


corpus = []

for f in filepaths:
    corpus.append(preproccesing(f))


def index_json(corpus):
    d = {}
    for i, text in enumerate(corpus):
        for word in text.split():
            if d.get(word) is None:
                d[word] = []
            d[word].append(i)

    data = {}
    for key, value in d.items():
        data[key] = len(value), list(set(value))

    with open('index.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


index_json(corpus)

with open('index.json', encoding='utf-8') as f:
    data = json.load(f)

friends_names = ['моника', 'рэйчел', 'фиби', 'росс', 'чендлер', 'джо']

friends = {}
for name in friends_names:
    friends[name] = data[name][0]

friends = list(friends.items())
friends.sort(key=lambda i: i[1])
print('Самый статистически популярный герой: ', friends[-1])

list_d = list(data.items())
list_d.sort(key=lambda i: i[1])
print('Самое частотное слово: ', list_d[-1][0], list_d[-1][1][0])

most_rarest = []
words_all_docs = []
for key, value in data.items():
    if value[0] == 1:
        most_rarest.append(key)
    if len(value[1]) == num_episodes:
        words_all_docs.append(key)
print('Самые редкие слова (встречаются 1 раз): ', most_rarest)
print('Слова, которые есть во всех документах: ', words_all_docs)


def index_matrix(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    with open('matrix.pkl', 'wb') as f:
        pkl.dump(X.toarray(), f)


index_matrix(corpus)
