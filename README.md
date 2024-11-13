# Course: information retrieval

[Project](https://github.com/AnnMokhova/infosearch/tree/master/infosearch_final) with implementation of a search engine on the webpage, where a user can choose one of the 4 search methods: tf-idf, bm25, word2vec generated through matrix, and word2vec generated through vectros. 

`corpus.csv` -- [collection of texts](https://docs.google.com/spreadsheets/d/1TfgKZktUPwXrAv3sobuRa9WW7Ctgk9GQT1Q4c8h_xjM/edit#gid=1224072571) which was used to build vectors and matrix

## To run application:
* run `create.py` to create vectors and matrix which will then be multiplied with the user's request, initialize vectorizers, models, etc.
* run `app.py`
