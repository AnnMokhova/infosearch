## Еще файлы для работы приложения:
- `corpus.csv` -- [коллекция](https://docs.google.com/spreadsheets/d/1TfgKZktUPwXrAv3sobuRa9WW7Ctgk9GQT1Q4c8h_xjM/edit#gid=1224072571) текстов, по которым строились матрицы и векторы
- `matrix_tfidf.pkl` -- [матрица тф-идф](https://drive.google.com/file/d/132VBq_pSF1ZJ5Fufr-3jIRYbksfwXjyI/view?usp=sharing)
- `matrix_w2v.pkl` -- [матрица вордтувек](https://drive.google.com/file/d/1J1AxBxPsiw4eQTIinCI7i3TwsTvyOv0m/view?usp=sharing)
- разархивированная папка `[araneum_none_fasttextcbow_300_5_2018.tgz]`(http://rusvectores.org/static/models/rusvectores4/fasttext/araneum_none_fasttextcbow_300_5_2018.tgz)

## Чтобы работало:
* сначала надо запустить `create.py`, чтобы создались матрицы и векторы, которые будут потом умножаться с запросом, инициализировались вейторайзеры, модели и т.д.  
* потом `app.py`
