# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:30:46 2017

@author: Administrator
"""

import re
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse

from my_tokenizer import MyTokenizer

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_files

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Path dos ficheiros pickle de teste e treino
train_fp = "../resources/train_db.p"
test_fp = "../resources/test_db.p"


def make_pickle_db():
    """Carrega os ficheiros texto das reviews, cria um dicionario
    com a informacao e guarda esse dicionario num ficheiro pickle
    """
    # Carregamento das reviews de treino num dicionario
    trainDic = load_files('../resources/aclImdb/train/')
    # Gravação do dicionario num ficheiro pickle
    pickle.dump(trainDic, open("../resources/train_db.p", "wb"))
    # Carregamento das reviews de teste num dicionario
    testDic = load_files('../resources/aclImdb/test/')
    # Gravação do dicionario num ficheiro pickle
    pickle.dump(testDic, open("../resources/test_db.p", "wb"))

# Ficheiros Treino
# Carregar o ficheiro pickle
train_db = pickle.load(open(train_fp, "rb"))
# Guardar array dos textos
train_text = train_db.data
# Guardar array com o indice da classe a que cada texto pertence
train_class = train_db.target

# Ficheiros Teste
# Carregar o ficheiro pickle
test_db = pickle.load(open(test_fp, "rb"))
# Guardar array dos textos
test_text = test_db.data
# Guardar array com o indice da classe a que cada texto pertence
test_class = test_db.target

# Limpeza dos documentos texto
# Remove tudo que não sejam caracteres do alfabeto
train_text = [re.sub(r'[^a-zA-Z]+', ' ', doc) for doc in train_text]
train_text = [re.sub(r'\b[A-Za-z]\w{0,3}\b', '', doc) for doc in train_text]

# Remove tudo que não sejam caracteres do alfabeto
test_text = [re.sub(r'[^a-zA-Z]+', ' ', doc) for doc in test_text]
test_text = [re.sub(r'\b[A-Za-z]\w{0,3}\b', '', doc) for doc in test_text]

""" Instanciar o objeto TfidfVectorizer com min_df=5 para remover as palavras
que ocorrem menos de 7 vezes e remove "stop words" (palavras vazias) do
vocabulario ingles.
"""

# alinea a


def log_reg_class():
    print('log_reg')
    """ Aferir o desempenho do discriminante logistico
    """
    # Tuplo com o numero maximo de vocabulario usado em cada teste
    mx_feat = (1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500,
               6000, 6500)
    # Dicionario onde serão guardados os resultados
    desempenho = {}
    for mf in mx_feat:
        tfidf = TfidfVectorizer(min_df=7, stop_words="english",
                                tokenizer=MyTokenizer(), max_features=mf)
        tfidf.fit(train_text)
        # Transformacao dos texto em vetores (Bag of Words)
        train_bow = tfidf.transform(train_text)
        # Transformacao dos texto em vetores (Bag of Words)
        test_bow = tfidf.transform(test_text)
        # Classificacao usando regressão
        logreg = LogisticRegression()
        logreg.fit(train_bow, train_class)
        result = logreg.predict(test_bow)
        confmtrx = confusion_matrix(test_class, result)
        fscore = f1_score(test_class, result)
        mean_error = mean_squared_error(test_class, result)
        values = {"result": result, "confmtrx": confmtrx, "fscore": fscore,
                  "mean_sqr_error": mean_error}
        desempenho.update({mf: values})
        print(mf)
    return values


def knn_k_ideal():
    print('knn_k_ideal')
    # alinea b
    tfidf = TfidfVectorizer(min_df=7, stop_words="english",
                            tokenizer=MyTokenizer(), max_features=4)
    tfidf.fit(train_text)
    train_bow = tfidf.transform(train_text)
    test_bow = tfidf.transform(test_text)
    neighbor_score_eucli = np.zeros(10)
    # Calcular o numero de vizinhos ideal para a distancia euclidiana
    for n in range(1, 7):
        knn = KNeighborsClassifier(n_neighbors=n, metric="euclidean",
                                   weights='uniform')
        knn.fit(train_bow[:40], train_class[:40])
        result = knn.predict(test_bow[:40])
        neighbor_score_eucli[n-1] = f1_score(test_class[:40], result)
        plt.plot(neighbor_score_eucli)

    # Calcular o numero de vizinhos ideal para a distancia de coseno
    train_bow_n = np.sqrt(np.sum(train_bow.toarray()**2, axis=1))
    test_bow_n = np.sqrt(np.sum(test_bow.toarray()**2, axis=1))
    neighbor_score_cos = np.zeros(10)
    for n in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=n, metric="euclidean",
                                   weights='uniform')
        knn.fit(train_bow_n[:40], train_class[:40])
        result = knn.predict(test_bow_n[:40])
        neighbor_score_cos[n-1] = f1_score(test_class[:40], result)
        plt.plot(neighbor_score_cos)
    return (neighbor_score_eucli, neighbor_score_cos)


def knn():
    print('knn')
    # Dados não processados
    tfidf = TfidfVectorizer(min_df=7, stop_words="english",
                            tokenizer=MyTokenizer(), max_features=400)
    tfidf.fit(train_text)
    train_bow = tfidf.transform(train_text)
    test_bow = tfidf.transform(test_text)

    desempenho = {}

    # Calculo com distancia eclidiana
    knn_eucl = KNeighborsClassifier(n_neighbors=3, metric="euclidean",
                                    weights='uniform')
    knn_eucl.fit(train_bow, train_class)
    result_eucl = np.zeros(25000)
    for i in range(0, 20001, 5000):
        result_eucl[i, i + 5000] = knn_eucl.predict(test_bow[i, i + 5000])
    confmtrx = confusion_matrix(test_class, result_eucl)
    fscore = f1_score(test_class, result_eucl)
    mean_error = mean_squared_error(test_class, result_eucl)
    values = {"result": result_eucl, "confmtrx": confmtrx, "fscore": fscore,
              "mean_sqr_error": mean_error}
    desempenho.update({'dist_eucli': values})

    # Calculo com distancia de cosseno
    train_bow_n = np.sqrt(np.sum(train_bow**2, axis=1))
    test_bow_n = np.sqrt(np.sum(test_bow**2, axis=1))
    knn_cos = KNeighborsClassifier(n_neighbors=3, metric="euclidean",
                                   weights='uniform')
    knn_cos.fit(train_bow_n, train_class)
    result_cos = np.zeros(25000)
    for i in range(0, 20001, 5000):
        result_cos[i, i + 5000] = knn_cos.predict(test_bow_n[i, i + 5000])
    confmtrx = confusion_matrix(test_class, result_cos)
    fscore = f1_score(test_class, result_cos)
    mean_error = mean_squared_error(test_class, result_cos)
    values = {"result": result_cos, "confmtrx": confmtrx, "fscore": fscore,
              "mean_sqr_error": mean_error}
    desempenho.update({'dist_cos': values})

    # Dados pre-processados
    test_bow_mx = np.mean(test_bow[:400], axis=1)
    test_bow_pp = test_bow[:400] - test_bow_mx
    train_bow_mx = np.mean(train_bow, axis=1)
    train_bow_pp = train_bow - train_bow_mx

    # Calculo com distancia eclidiana
    knn_eucl.fit(train_bow_pp, train_class)
    result_pp_eucl = np.zeros(25000)
    for i in range(0, 20001, 5000):
        result_pp_eucl[i, i + 5000] = knn_eucl.predict(
                result_pp_eucl[i, i + 5000])
    confmtrx = confusion_matrix(test_class, result_pp_eucl)
    fscore = f1_score(test_class, result_pp_eucl)
    mean_error = mean_squared_error(test_class, result_pp_eucl)
    values = {"result": result_pp_eucl, "confmtrx": confmtrx, "fscore": fscore,
              "mean_sqr_error": mean_error}
    desempenho.update({'dist_eucli_pp': values})

    # Calculo com distancia de cosseno
    test_bow_pp_n = np.sqrt(np.sum(test_bow_pp**2, axis=1))
    train_bow_pp_n = np.sqrt(np.sum(train_bow_pp**2, axis=1))
    knn_cos.fit(train_bow_pp_n, train_class)
    result_pp_cos = np.zeros(25000)
    for i in range(0, 20001, 5000):
        result_pp_cos[i, i + 5000] = knn_cos.predict(
                test_bow_pp_n[i, i + 5000])
    confmtrx = confusion_matrix(test_class, result_pp_cos)
    fscore = f1_score(test_class, result_pp_cos)
    mean_error = mean_squared_error(test_class, result_pp_cos)
    values = {"result": result_pp_cos, "confmtrx": confmtrx, "fscore": fscore,
              "mean_sqr_error": mean_error}
    desempenho.update({'dist_cos_pp': values})

    return desempenho


def classificador(predited_values):
    class_values = predited_values.copy()
    class_values[predited_values < 1] = 1
    class_values[predited_values > 10] = 10
    for i in range(2, 11):
        condition_a = predited_values < i
        condition_b = predited_values >= i-1
        class_values[np.all((condition_a, condition_b), axis=0)] = i
    return class_values


# alinea 2
def lin_reg_class_sklearn():
    """Regressor Linear sklearn 
    """
    test_score_class = np.array([int(re.sub(r'.+?(_)', '', filename)[:-4])
            for filename in test_db.filenames])
    train_score_class = np.array([int(re.sub(r'.+?(_)', '', filename)[:-4])
            for filename in train_db.filenames])

    # conversao do teste de treino para Bag of Words
    mf = 400
    tfidf = TfidfVectorizer(min_df=7, stop_words="english",
                            tokenizer=MyTokenizer(), max_features=mf)
    tfidf.fit(train_text)
    # Transformacao dos texto em vetores (Bag of Words)
    train_bow = tfidf.transform(train_text)
    # Transformacao dos texto em vetores (Bag of Words)
    test_bow = tfidf.transform(test_text)

    lin_reg = LinearRegression()
    desempenho = {}

    # Testar Classificador com 10 classes
    lin_reg.fit(test_bow, test_score_class)
    c10_result = lin_reg.predict(train_bow)
    lin_reg.score(train_bow, train_score_class)

    c10_confmtrx = confusion_matrix(train_score_class, c10_result)
    c10_fscore = f1_score(train_score_class, c10_result)
    c10_mean_error = mean_squared_error(train_score_class, c10_result)

    c10_values = {"result": c10_result, "confmtrx": c10_confmtrx,
                  "fscore": c10_fscore, "mean_sqr_error": c10_mean_error}
    desempenho.update({'sklearn_c10': c10_values})

    # Testar Classidicador com 2 classes
    lin_reg.fit(test_bow, test_class)
    c2_result = lin_reg.predict(train_bow)
    c2_confmtrx = confusion_matrix(train_bow, c2_result)
    c2_fscore = f1_score(train_score_class, c2_result)
    c2_mean_error = mean_squared_error(train_score_class, c2_result)

    c2_values = {"result": c2_result, "confmtrx": c2_confmtrx,
                 "fscore": c2_fscore, "mean_sqr_error": c2_mean_error}
    desempenho.update({'sklearn_c2': c2_values})

    return desempenho

def lin_reg_class_handmade():
    """Regressor linear calculado
    """
    test_score_class = np.array([int(re.sub(r'.+?(_)', '', filename)[:-4])
            for filename in test_db.filenames])
    train_score_class = np.array([int(re.sub(r'.+?(_)', '', filename)[:-4])
            for filename in train_db.filenames])

    # conversao do teste de treino para Bag of Words
    mf = 400
    tfidf = TfidfVectorizer(min_df=7, stop_words="english",
                            tokenizer=MyTokenizer(), max_features=mf)
    tfidf.fit(train_text)
    # Transformacao dos texto em vetores (Bag of Words)
    train_bow = tfidf.transform(train_text)
    # Transformacao dos texto em vetores (Bag of Words)
    test_bow = tfidf.transform(test_text)

    # adicionar a linha de uns
    ones = sparse.csc_matrix(np.ones(25000))
    train_X = sparse.vstack([ones, train_bow.T])

    # defenir as saidas desejadas
    Y = -np.ones((10, train_bow.shape[0]))
    for i in range(train_bow.shape[0]):
        Y[test_score_class[i] - 1, i] = 1
    Y = sparse.csc_matrix(Y)

    # calculo ma matriz de transformacao
    Rx = train_X.dot(train_X.T)
    rxy = train_X.dot(Y.T)
    Rxi = sparse.csc_matrix(np.linalg.pinv(np.array(Rx.toarray())))
    w = Rxi.dot(rxy)

    # conversao dos dados de teste para BoW
    test_X = sparse.vstack([ones, test_bow.T])

    desempenho = {}

    Yh = w.T.dot(test_X)

    c10_result = np.argmax(Yh.toarray(), axis=0) + 1
    c10_confmtrx = confusion_matrix(train_score_class, c10_result)
    c10_mean_error = mean_squared_error(train_score_class, c10_result)
    c10_values = {"result": c10_result, "confmtrx": c10_confmtrx,
                  "mean_sqr_error": c10_mean_error}
    desempenho.update({'sklearn_c10': c10_values})

    c2_result = c10_result > 5
    c2_confmtrx = confusion_matrix(train_class, c2_result)
    c2_fscore = f1_score(train_class, c2_result)
    c2_mean_error = mean_squared_error(train_class, c2_result)

    c2_values = {"result": c2_result, "confmtrx": c2_confmtrx,
                 "fscore": c2_fscore, "mean_sqr_error": c2_mean_error}
    desempenho.update({'sklearn_c2': c2_values})

    return desempenho