"""
file: train_w2vec.py
author: Mary St. Jean
Adapted from https://github.com/3778/icd-prediction-mimic/tree/master?tab=readme-ov-file
Consulted https://www.kaggle.com/code/jagannathrk/word2vec-cnn-text-classification
"""

import argparse
import datasets
from constants import DATA_DIR
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models.word2vec import Word2Vec
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#import gensim
#from nltk.tokenize import sent_tokenize, word_tokenize
#import warnings



def main(args, verbose = 1):
    # Load dataset
    nbme = datasets.NBME_Dataset()
    nbme.load_preprocessed()
    nbme.split()



    # Instantiate embedding

    notes = []
    for i in nbme.df_x['pn_history']:
        notes.append(i.split())


    model = Word2Vec(notes, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")


    token = Tokenizer(127723)
    token.fit_on_texts(nbme.df_x['pn_history'])
    text = token.texts_to_sequences(nbme.df_x['pn_history'])
    text = pad_sequences(text, 75)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(nbme.df_y)
    nbme.df_y['feature_num'] = to_categorical(y)

    X = nbme.df_x
    X['embeddings'] = np.array(text).tolist()


    X_train, X_test, y_train, y_test = train_test_split(X, nbme.df_y, test_size = 0.33, random_state = 42)

    print(X_train.head())

    nbme.save_procesed(X_train,path=DATA_DIR, fn='x_train')
    nbme.save_procesed(X_test, path=DATA_DIR, fn='x_test')
    nbme.save_procesed(y_train,path=DATA_DIR, fn='y_train')
    nbme.save_procesed(y_test, path=DATA_DIR, fn='y_test')

    if verbose:
        print(f'''
            Data Split: {X_train.shape[0]}, {X_test.shape[0]}
            ''')

    print(f'''
        Word2Vec embeddings saved!
    ''')






def arg_parser():
    parser = argparse.ArgumentParser(description='Train Word2Vec word embeddings')
    parser.add_argument('-workers', type=int, dest='workers', default=8, help='Number of CPU threads for W2V training.')
    parser.add_argument('--reset_stopwords', type=bool, dest='reset_stopwords', default=0,
                        help='True to set stopwords vectors to null. Default False.')
    parser.add_argument('--train_method', type=bool, dest='sg', default=1,
                        help='W2V train method. 0 for CBoW, 1 for Skipgram.')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()

    main(args)