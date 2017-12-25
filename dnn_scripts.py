import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
# import tensorflow as tf
import pickle
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr
import argparse
# import keras
# from keras.models import Sequential, Model, model_from_json
# from keras.layers import Input, Embedding, LSTM, Dense
# from keras.layers.core import Reshape, Flatten, Dropout
# from keras import metrics, optimizers
import re
import datetime
import os.path
import random
import matplotlib.pyplot as plt
import re

data_dir = 'data'
MEALS_LABELS = ['Weight', 'Protein_g', 'TotalLipid_g', 'Carbohydrate_g', 'Water_g', 'Alcohol_g', 'Energy_kcal']
HOME_DIR = '/home/noamba/dnnChallenge/'
SRC_DIR = '/home/noamba/dnnChallenge/src/'

# create the raw xy dataframe
def create_raw_glucose():
    glucose_df = pd.read_pickle(os.path.join(data_dir, 'GlucoseValues.df'))

    def resample_new(x):
        return x.reset_index(level=['ConnectionID'], drop=True). \
            resample(str(5) + 'min', label='right').last().ffill()

    ground = glucose_df.sort_index().groupby(level='ConnectionID').apply(resample_new)
    g144 = ground.groupby(level='ConnectionID').apply(lambda group: group.iloc[145:, ]).reset_index(level=1, drop=True)
    g12 = g144.sort_index(axis=0, ascending=False).diff(12).sort_index(axis=0)
    gtot = g12.groupby(level='ConnectionID').apply(lambda group: group.iloc[:-13, ]).reset_index(level=1, drop=True)
    gtot.columns = ['label']
    gtot.to_pickle('/home/noamba/dnnChallenge/src/data/x_y_raw_minus.df')
    ggg = gtot.copy()
    ggg['label'] = ggg['label'].apply(lambda x: x*-1)
    ggg.to_pickle('/home/noamba/dnnChallenge/src/data/x_y_raw_fixed.df')
    ggg.to_pickle('/home/noamba/dnnChallenge/src/data/x_y_raw.df')


def HW5():
    with open('/home/noamba/dnnChallenge/feature_test_full.pickle', 'rb') as handle:
        ft = pickle.load(handle)
    labels1 = ['Blood tests', 'Measurements', 'Bac PCs', 'Glucose means', 'Glucose max', 'Glucose linear fit', 'Time of day',
             'Meals', 'Raw glucose values', 'Using all features']
    labels2 = ['Blood tests', 'Measurements', 'Bac PCs', 'Glucose means', 'Glucose max', 'Glucose linear fit', 'Time of day',
             'Meals', 'Raw glucose values', 'All features set to zero']
    vals1 = ft[1][0:len(labels1)]
    vals2 = ft[1][len(labels1):]
    r1 = np.array(ft[2][0:len(labels1)])[:,0]
    r2 = np.array(ft[2][len(labels1):])[:,0]
    xvals = range(len(vals1))

    plt.figure(1)
    plt.clf()
    plt.barh(xvals, vals1, color='red')
    plt.yticks(np.array(xvals)+0.3, labels1)
    plt.xlabel("Squared loss")
    for i in range(len(vals1)):
        plt.text(vals1[i]+5, i+0.2, str(round(vals1[i], 2)))
    plt.title("Squared loss when using only the specified feature")
    plt.savefig(HOME_DIR + 'feature_test_using_only_specified_feature.png', bbox_inches='tight')

    plt.figure(2)
    plt.clf()
    plt.barh(xvals, vals2, color='green')
    plt.yticks(np.array(xvals)+0.3, labels2)
    plt.xlabel("Squared loss")
    for i in range(len(vals2)):
        plt.text(vals2[i]+5, i+0.2, str(round(vals2[i], 2)))
    plt.title("Squared loss when setting specified feature to zero")
    plt.savefig(HOME_DIR + 'feature_test_setting_specified_feature_to_0.png', bbox_inches='tight')

    plt.figure(3)
    plt.clf()
    plt.barh(xvals, r1, color='red')
    plt.yticks(np.array(xvals)+0.3, labels1)
    plt.xlabel("Pearson correlation coef.")
    for i in range(len(r1)):
        plt.text(r1[i], i+0.2, str(round(r1[i], 3)))
    plt.title("Pearson correlation coef. when using only the specified feature")
    plt.savefig(HOME_DIR + 'feature_test_using_only_specified_feature_cor.png', bbox_inches='tight')

    plt.figure(4)
    plt.clf()
    plt.barh(xvals, r2, color='green')
    plt.yticks(np.array(xvals)+0.3, labels2)
    plt.xlabel("Pearson correlation coef.")
    for i in range(len(r2)):
        plt.text(r2[i], i+0.2, str(round(r2[i], 3)))
    plt.title("Pearson correlation coef. when setting specified feature to zero")
    plt.savefig(HOME_DIR + 'feature_test_setting_specified_feature_to_0_cor.png', bbox_inches='tight')

def main():
    HW5()
    return

main()