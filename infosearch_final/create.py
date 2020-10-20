import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
from gensim.models import KeyedVectors

df = pd.read_csv('corpus.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]


# словарь, по ключам которого будет находиться самого релевантного документа
document_response = {}
n = 0
for index, row in df.iterrows():
    document_response[n] = row['Текст вопроса']
    n += 1

# создаем и сохраняем матрицу тф-идф
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Текст с препроцессингом']).toarray()

with open('matrix_tfidf.pkl', 'wb') as f:
    pkl.dump(X, f)

# создаем и сохраняем векторы для каждого документа коллекции
model_file = 'araneum_none_fasttextcbow_300_5_2018.model'
model = KeyedVectors.load(model_file)


def normalize_vec(v):
    return v / np.sqrt(np.sum(v ** 2))


def word2vec_vector(text: str):
    lemmas = text.split()
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    vec = np.zeros((model.vector_size,))
    # если слово есть в модели, берем его вектор
    for idx, lemma in enumerate(lemmas):
        if lemma in model:
            lemmas_vectors[idx] = model[lemma]

    # проверка на случай, если на вход пришел пустой массив
    if lemmas_vectors.shape[0] is not 0:
        vec = np.mean(lemmas_vectors, axis=0)

    return vec


corpus_vectors = np.array([normalize_vec(word2vec_vector(text)) for text in df['Текст с препроцессингом']])
with open('vectors_w2v.pkl', 'wb') as f:
    pkl.dump(corpus_vectors, f)


# создаем и сохраняем матрицы для каждого документа коллекции
def word2vec_matrix(text):
    lemmas = text.split()
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    vec = np.zeros((model.vector_size,))

    for idx, lemma in enumerate(lemmas):
        if lemma in model.wv:
            lemmas_vectors[idx] = normalize_vec(model.wv[lemma])

    return lemmas_vectors


corpus_matrix = np.array([word2vec_matrix(text) for text in df['Текст с препроцессингом']])

with open('matrix_w2v.pkl', 'wb') as f:
    pkl.dump(corpus_matrix, f)
