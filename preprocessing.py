"""
file: preprocessing.py
author: Mary St. Jean
Adapted from https://github.com/3778/icd-prediction-mimic/tree/master?tab=readme-ov-file
"""

import datasets

def main():

    nbme = datasets.NBME_Dataset()

    nbme.preprocess()

    nbme.save_preprocessed()


if __name__ == '__main__':

    main()