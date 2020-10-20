from flask import Flask, request, render_template
from create import vectorizer, document_response, normalize_vec, word2vec_vector, word2vec_matrix, df
import pickle as pkl
from typing import Any
from razdel import tokenize
import pymorphy2
import numpy as np
from rank_bm25 import BM25Okapi

app = Flask(__name__)

corpus = df['Текст с препроцессингом']

morph = pymorphy2.MorphAnalyzer()


# для обработки входящего запроса
def preproccesing(sentence):
    tokens = [_.text for _ in tokenize(sentence) if _.text not in '1234567890,.!?-();:""«»—_–#' and '.' not in _.text]
    return ' '.join([morph.parse(t)[0].normal_form for t in tokens])


def vector_tfidf(query: str, vectorizer: Any) -> Any:
    new_query = []
    for word in query.split():
        if word in vectorizer.vocabulary_.keys():
            new_query.append(word)
    if len(new_query) == 0:
        raise ValueError
    return vectorizer.transform(new_query).toarray()


def ranging(res):
    d = {}
    # создаем словарь, где ключ - индекс документа, значение - близость документа с запросом
    index = 0
    for doc in res:
        d[index] = doc
        index += 1
    sorted_by_value = sorted(d.items(), key=lambda x: x[1], reverse=True)

    return sorted_by_value[0][0]


def tfidf(query):
    with open('matrix_tfidf.pkl', 'rb') as f:
        X = pkl.load(f)
    vector_query = vector_tfidf(preproccesing(query), vectorizer)
    res_tfidf = np.dot(X, vector_query[0]).tolist()
    return document_response[ranging(res_tfidf)]


def bm_25(query):
    bm25 = BM25Okapi(corpus)
    best_doc = bm25.get_top_n(preproccesing(query), corpus, n=1)[0]
    best_idx = corpus.tolist().index(best_doc)
    return document_response[best_idx]


def w2v_vector(q):
    query = normalize_vec(word2vec_vector(preproccesing(q)))
    with open('vectors_w2v.pkl', 'rb') as f:
        vectors = pkl.load(f)
    res = np.dot(vectors, np.array(query)).tolist()
    best_doc = ranging(res)
    return document_response[best_doc]


def w2v_matrix(q, reduce_func=np.max, axis=0):
    query = word2vec_matrix(preproccesing(q))
    with open('matrix_w2v.pkl', 'rb') as f:
        docs = pkl.load(f)
    sims = []
    for doc in docs:
        sim = doc.dot(query.T)
        sim = reduce_func(sim, axis=axis)
        sims.append(sim.sum())
    best_doc = np.argmax(sims)
    return document_response[best_doc]


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":
        selected = request.form['method']
        text = str(request.form['text'])
        if selected == 'tfidf':
            return tfidf(text)
        elif selected == 'bm25':
            return bm_25(text)
        elif selected == 'w2v_vector':
            return w2v_vector(text)
        elif selected == 'w2v_matrix':
            return w2v_matrix(text)

    return render_template('main.html')


if __name__ == '__main__':
    app.run(debug=True)
