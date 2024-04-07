"""
file: data_sets.py
author: Mary St. Jean
Adapted from https://github.com/3778/icd-prediction-mimic/tree/master?tab=readme-ov-file
"""


import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MultiLabelBinarizer

from constants import DATA_DIR

import utils


class NBME_Dataset:

    def __init__(self):
        self.name = 'NBME'

    def load_preprocessed(self, path=DATA_DIR):
        with open(f'{path}nbme_data.pkl', 'rb') as file:
            self.df = pickle.load(file)

    def save_preprocessed(self, path=DATA_DIR):
        pd.to_pickle(self.df, f'{path}nbme_data.pkl')

    def save_procesed(self, df, fn, path=DATA_DIR):
        pd.to_pickle(df, f'{path}{fn}.pkl')

    def preprocess(self, verbose=1):

        df_text = (pd.read_csv(f'{DATA_DIR}patient_notes.csv'))

        df_features = (pd.read_csv(f'{DATA_DIR}train.csv'))

        #df_mapping = (pd.read_csv(f'{DATA_DIR}train.csv'))

        self.df = pd.merge(df_features, df_text, on=['pn_num'], how='outer')

        if verbose:
            print(f'''
            -------------
            Total unique features: {self.df.feature_num.explode().nunique()}
            Total samples: {self.df.shape[0]}
            Data preprocessed!
            ''')

    def split(self):

        # Load ordered list of features
        self.all_features = (pd.read_csv(f'{DATA_DIR}train.csv'))

        self.df_x = self.df[['case_num_y','pn_num','annotation','pn_history']]
        self.df_y = self.df[['feature_num']]




















