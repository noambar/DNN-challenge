import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
import pickle
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr
import argparse
import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Embedding, LSTM, Dense
from keras.layers.core import Reshape, Flatten, Dropout
from keras import metrics, optimizers
import re
import datetime
from sklearn import linear_model
# from keras import backend as K

# The time series that you would get are such that the difference
# between two rows is 15 minutes. This is a global number that we
# used to prepare the data, so you would need it for different purposes
DATA_RESOLUTION_MIN = 5
# DATA_RESOLUTION_MIN = 15
SAMPLES_PER_HOUR = int(60/DATA_RESOLUTION_MIN)
HOURS_BACKWARD = 12
MEASUREMENTS_LABELS = ['BMI', 'BodyWeight', 'HeartRate', 'Height', 'Hips', 'Waist', 'Age']
BLOOD_LABELS = ['Cholesterol, total', 'HDL Cholesterol', 'HbA1C%']
MEALS_LABELS = ['Weight', 'Protein_g', 'TotalLipid_g', 'Carbohydrate_g', 'Water_g', 'Alcohol_g', 'Energy_kcal']
EXERCISES_LABELS = ['Exercise', 'Intensity']
HOME_DIR = '/home/noamba/dnnChallenge/'
SRC_DIR = '/home/noamba/dnnChallenge/src/'
# Some model parameters
EPOCHS = 15
CONV_xD = 2
KERNEL_SIZE = 5
FILTERS = [64, 64, 64, 64]
STRIDES = [2, 2, 2, 2]
STRIDES2D = [(2,2), (2,2), (2,2), (2,2)]
LEARNING_RATE = 0.001
TRAIN_PART = 0.90
BATCH_SIZE = 100

# DAY_or_NIGHT = 'NIGHT'
# DAY_or_NIGHT = 'DAY'
DAY_or_NIGHT = ''

# NORM = 'NORM'
NORM = ''

MODEL_PATH = SRC_DIR + 'trained_model_' + str(CONV_xD) +'d_' + str(EPOCHS) + 'e_' + str(DATA_RESOLUTION_MIN) + \
             'min_' +  str(KERNEL_SIZE) + 'k_' + str(TRAIN_PART) + 'tp_' + str(LEARNING_RATE) + 'lr_' + DAY_or_NIGHT + NORM + '.h5'

MODEL_PARAM = [MODEL_PATH, EPOCHS, CONV_xD, KERNEL_SIZE, FILTERS, STRIDES, STRIDES2D, LEARNING_RATE, TRAIN_PART, BATCH_SIZE]

LOG_FILE = '/home/noamba/dnnChallenge/log_file.log'
MODEL_LOG_FILE = '/home/noamba/dnnChallenge/.log_models.log'

MODELS_TO_USE = ['/home/noamba/dnnChallenge/src/trained_model_2d_16e_5min.h5',
                 '/home/noamba/dnnChallenge/src/trained_model_2d_15e_5min_6k_0.9tp_0.001lr_.h5',
                 '/home/noamba/dnnChallenge/src/trained_model_1d_20e_5min_5k_0.9tp_0.001lr_.h5',
                 '/home/noamba/dnnChallenge/src/trained_model_1d_20e_5min_7k_0.9tp_0.001lr_.h5',
                 '/home/noamba/dnnChallenge/src/trained_model_1d_22e_5min_5k_0.9tp_0.001lr_.h5',
                 '/home/noamba/dnnChallenge/src/trained_model_1d_25e_5min_5k_0.9tp_0.002lr_.h5',
                 '/home/noamba/dnnChallenge/src/trained_model_2d_15e_5min_5k_0.9tp_0.001lr_.h5',
                 '/home/noamba/dnnChallenge/src/trained_model_1d_25e_5min_5k_0.9tp_0.0005lr_.h5',
                 '/home/noamba/dnnChallenge/src/trained_model_1d_20e_5min_8k_0.9tp_0.001lr_.h5',
                 '/home/noamba/dnnChallenge/src/trained_model_2d_20e_5min_4k_0.9tp_0.001lr_.h5',
                 '/home/noamba/dnnChallenge/src/trained_model_2d_20e_5min_5k_0.95tp_0.001lr_.h5',
                 '/home/noamba/dnnChallenge/src/trained_model_1d_20e_5min_5k_0.75tp_0.001lr_.h5',
                 '/home/noamba/dnnChallenge/src/trained_model_2d_15e_5min_6k_0.75tp_0.001lr_.h5',
                 '/home/noamba/dnnChallenge/src/trained_model_1d_30e_5min_5k_0.99tp_0.001lr_.h5']


NIGHT_MODELS = ['/home/noamba/dnnChallenge/src/trained_model_2d_15e_5min_5k_0.9tp_0.001lr_NIGHT.h5',
                '/home/noamba/dnnChallenge/src/trained_model_1d_15e_5min_5k_0.9tp_0.001lr_NIGHT.h5']

# MODELS_TO_USE = ['/home/noamba/dnnChallenge/src/trained_model_2d_15e_5min_5k_0.9tp_0.001lr_NORM.h5']
# NIGHT_MODELS = []


class DFMeta(object):
    """
    A class to hold meta data about the different dataframes
    """
    def __init__(self, fname, ts_field_name):
        """
        :param fname: file name of the dataframe
        :param ts_field_name: column name for "timestamp" in this dataframe
        """
        self.fname = fname
        self.ts_field_name = ts_field_name


class GlucoseNetwork:
    def __init__(self, features_size, model_path):
        """ Initialize a neural network predictor object.
        """
        self.model_path = model_path
        self.model = None
        self.features_size = features_size
        



    def init_graph(self):
        # TODO in the future change to different options of initialization, given in __init__
        """ get the network basic graph (before training
            input : vector x s.t size of x = [1,features_size]
            output : scalar
            the net = 2 fc layers
            objective : minimize the euclidean loss
        """
        # create glucose input layer for convolution
        convolution_dim = HOURS_BACKWARD * SAMPLES_PER_HOUR + 1
        conv2d_dim = len(MEALS_LABELS) + len (EXERCISES_LABELS) + 1 + 1 # glucose + sleep
        raw_temporal_input = Input(shape=(convolution_dim, conv2d_dim, ), name='raw_input')

        # convolution layers

        if (CONV_xD == 2):
            x = Reshape((convolution_dim, conv2d_dim, ) + (1, ),
                        input_shape=(convolution_dim, conv2d_dim, ))(raw_temporal_input)
            conv1 = keras.layers.convolutional.Conv2D(filters=FILTERS[0], kernel_size=KERNEL_SIZE,
                                                      strides=STRIDES2D[0], padding='same', activation='sigmoid',
                                                      kernel_initializer='glorot_uniform')(x)
            conv2 = keras.layers.convolutional.Conv2D(filters=FILTERS[1], kernel_size=KERNEL_SIZE,
                                                      strides=STRIDES2D[1], padding='same', activation='sigmoid',
                                                      kernel_initializer='glorot_uniform')(conv1)
            conv3 = keras.layers.convolutional.Conv2D(filters=FILTERS[2], kernel_size=KERNEL_SIZE,
                                                      strides=STRIDES2D[2], padding='same', activation='sigmoid',
                                                      kernel_initializer='glorot_uniform')(conv2)
            conv4 = keras.layers.convolutional.Conv2D(filters=FILTERS[3], kernel_size=KERNEL_SIZE,
                                                      strides=STRIDES2D[3], padding='same', activation='sigmoid',
                                                      kernel_initializer='glorot_uniform')(conv3)
        elif (CONV_xD == 1):
            x = Reshape((convolution_dim, conv2d_dim, ) ,
                        input_shape=(convolution_dim, conv2d_dim, ))(raw_temporal_input)
            conv1 = keras.layers.convolutional.Conv1D(filters=FILTERS[0], kernel_size=KERNEL_SIZE,
                                                      strides=STRIDES[0], padding='same', activation='sigmoid',
                                                      kernel_initializer='glorot_uniform')(x)
            conv2 = keras.layers.convolutional.Conv1D(filters=FILTERS[1], kernel_size=KERNEL_SIZE,
                                                      strides=STRIDES[1], padding='same', activation='sigmoid',
                                                      kernel_initializer='glorot_uniform')(conv1)
            conv3 = keras.layers.convolutional.Conv1D(filters=FILTERS[2], kernel_size=KERNEL_SIZE,
                                                      strides=STRIDES[2], padding='same', activation='sigmoid',
                                                      kernel_initializer='glorot_uniform')(conv2)
            conv4 = keras.layers.convolutional.Conv1D(filters=FILTERS[3], kernel_size=KERNEL_SIZE,
                                                      strides=STRIDES[3], padding='same', activation='sigmoid',
                                                      kernel_initializer='glorot_uniform')(conv3)
        else:
            print ("you must choose between 1D and 2D convolution. ass")
            return

        flatten_cov = Flatten()(conv4)
        # a layer of dropout after the convolution
        dropout_conv = Dropout(0.2)(flatten_cov)
        dens_after_conv = Dense(10,  activation='sigmoid', kernel_initializer='glorot_normal')(dropout_conv)

        # other features are input to dense layers
        additional_features_input = Input(shape=(self.features_size- conv2d_dim*convolution_dim,), name='add_input')
        dens1 = Dense(int((self.features_size-conv2d_dim*convolution_dim)*1.1),  activation='sigmoid',
                   kernel_initializer='glorot_normal')(additional_features_input)
        dens2 = Dense(int((self.features_size-conv2d_dim*convolution_dim)*0.5),  activation='sigmoid',
                   kernel_initializer='glorot_normal')(dens1)
        # TODO: perhaps dropout should be on the input
        dropout1 = Dropout(0.2)(dens2)
        dens3 = Dense(2, activation='sigmoid', kernel_initializer='glorot_normal')(dropout1)

        concat1 = keras.layers.concatenate([dens_after_conv, dens3])
        main_output = Dense(1, kernel_initializer='glorot_normal', name='main_output')(concat1)
        self.model = Model(inputs=[raw_temporal_input, additional_features_input], outputs=[main_output])

    def train_network(self, x_all, y_all, train_part, epochs, batch_size):
        """ Train the network with the given data
        """
        train_X_raw, train_X_other, train_Y, val_X_raw, val_X_other, val_Y = \
            self.get_train_and_val(x_all, y_all, train_part)

        # initialize network architecture
        self.init_graph()

        # Compile model
        adam = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1)
        self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=[metrics.MSE])

        # Fit the model (on train data)
        hist = self.model.fit([train_X_raw, train_X_other], train_Y, epochs=epochs, batch_size=batch_size,
                              validation_data=([val_X_raw, val_X_other], val_Y))

        # save model:
        self.model.save(self.model_path)
        print(str(datetime.datetime.now()) + " - " + "model save to path:", self.model_path)


        # evaluate the model (train / val)
        scores_train = self.model.evaluate([train_X_raw, train_X_other], train_Y)
        scores_val = self.model.evaluate([val_X_raw, val_X_other], val_Y)
        print(str(datetime.datetime.now()) + " - " + "train score:", scores_train, ", validation score:", scores_val)
        with open(MODEL_LOG_FILE, 'a') as lf:
            lf.write(str(datetime.datetime.now()) + " - " + str(MODEL_PARAM) + " - " + "train score: " +
                     str(scores_train) + ", validation score: " + str(scores_val) + '\n')

        # del self.model
        return



    def get_train_and_val(self, x, y, train_part):
        """
        TODO Choose better ways to split to train and val - by people!
        :param x:
        :param y:
        :param train_size:
        :return:
        """
        # split to train and validation by ConnectionID
        xy_raw = x.filter(regex='_before_')
        temp_l = [xy_raw.filter(regex='Meals_' + comp + '_before') for comp in MEALS_LABELS]
        temp_l += [xy_raw.filter(regex='Exercises_' + comp + '_before') for comp in EXERCISES_LABELS]
        temp_l.append(xy_raw.filter(regex='Sleeps_slept_before'))
        temp_l.append(xy_raw.filter(regex='GlucoseValue_before'))

        xy_raw = np.dstack([com.values for com in temp_l])

        xy_other = x.select(lambda x: not re.search('_before_', x), axis=1)
        print(str(datetime.datetime.now()) + " - " + "raw feature's shape is: " + str(xy_raw.shape))
        print(str(datetime.datetime.now()) + " - " + "other feature's shape is: " + str(xy_other.shape))
        # xy_raw_values = xy_raw
        xy_other_values = xy_other.values
        cids = xy_other.index.get_level_values('ConnectionID').unique().tolist()
        train_size = round(len(cids) * train_part)
        train_cid_idx = np.random.choice(cids, size=train_size, replace=False)
        val_cids_idx = list(set(cids) - set(train_cid_idx))
        val_idx = np.where([q in val_cids_idx for q in xy_other.index.get_level_values('ConnectionID').tolist()])[0]
        train_idx = np.where([q in train_cid_idx for q in xy_other.index.get_level_values('ConnectionID').tolist()])[0]
        train_X_raw = xy_raw[train_idx, :, :]
        train_X_other = xy_other_values[train_idx, :]
        train_Y = y[train_idx]
        val_X_raw = xy_raw[val_idx, :, :]
        val_X_other = xy_other_values[val_idx, :]
        val_Y = y[val_idx]
        return train_X_raw, train_X_other, train_Y, val_X_raw, val_X_other, val_Y


    def predict_network(self, x):
        # load trained model
        self.model = keras.models.load_model(self.model_path)
        print(str(datetime.datetime.now()) + " - " + "[Predictor:predict_network] model loaded from " +
              self.model_path + "...")
        # predict y
        y = self.model.predict(x)
        return y


class Predictor(object):
    """
    This is where you should implement your predictor.
    The testing script calls the 'predict' function with X.
    You should implement this function as you wish, just not the function signature.
    The other functions are here just as an example for you to have something to start with.
    """

    def __init__(self, path2data): #DO NOT CHANGE THIS SIGNATURE
        """ Initialize predictor on the data in path2data
        :param path2data: path to the raw data frames
        """

        self.path = path2data
        # self.model_path = SRC_DIR + 'trained_model.h5'
        # self.model_path = SRC_DIR + 'trained_model_2d.h5'
        # self.model_path = SRC_DIR + 'trained_model_2d_' + str(EPOCHS) + 'e.h5'
        # self.model_path = SRC_DIR + 'trained_model_1d_' + str(EPOCHS) + 'e.h5'
        self.model_path = MODEL_PATH

        self.dfs_meta = {
            'glucose': DFMeta('GlucoseValues.df', 'Timestamp'),
            'exercises': DFMeta('Exercises.df', 'Timestamp'),
            'meals': DFMeta('Meals.df', 'Timestamp'),
            'sleep': DFMeta('Sleep.df', 'sleep_time'),
            'test_food': DFMeta('TestFoodsNew.df', 'Timestamp'),
            'bac': DFMeta('BacterialSpecies.df', None),
            'blood': DFMeta('BloodTests.df', None),
            'measurements': DFMeta('Measurements.df', None),
        }

        self.dfs = {}
        self.X = None
        self.load_raw_data()

    def load_raw_data(self):
        """ Loads raw dataframes from files, and does some basic cleaning """

        # Arrange sequential data: re-sample each 15 min
        def resample(x):
            return x.reset_index(level=['ConnectionID'], drop=True). \
                resample(str(DATA_RESOLUTION_MIN) + 'min', label='right').last().ffill()

        for df_name, dfmeta in self.dfs_meta.items():
            fname = dfmeta.fname
            path2df = os.path.join(self.path, fname)
            self.dfs[df_name] = pd.read_pickle(path2df).sort_index()

        # fix glucose index names
        self.dfs['glucose'].index.names = ['ConnectionID', 'Timestamp']

        # resample glucose indices for DATA_RESOLUTION_MIN min:
        self.dfs['glucose'] = self.dfs['glucose'].sort_index().groupby(level='ConnectionID').apply(resample)

    # ------------------------------------- features member functions -------------------------------------

    def get_time_features(self, X):
        """
        Get the time of day 
        :param X: 
        :return: 
        """
        print(str(datetime.datetime.now()) + " - " + "[Predictor:get_time_fatures] start building time features...")
        tmp_x = X.copy()
        tmp_x['time'] = tmp_x.index.get_level_values('Timestamp')
        tmp_x.time = tmp_x.time.apply(lambda t: t.hour + (t.minute / 60))
        return [tmp_x.time]

    def get_raw_temporal_features(self, X):
        """
        Get all temporal features
        :param X:
        :return:
        """
        # with open(LOG_FILE, 'a') as lf:
        #     lf.write(str(datetime.datetime.now()) + " " + MODEL_PATH + '\n')
        #     lf.write(str(datetime.datetime.now()) + ' meals' + '\n')

        meals_df = self.get_raw_meals(X)
        meals_df.columns = ['Meals_' + comp for comp in meals_df.columns]


        # with open(LOG_FILE, 'a') as lf:
        #     lf.write(str(datetime.datetime.now()) + ' sleep' + '\n')



        sleeps_df = self.get_raw_sleep(X)
        sleeps_df.columns = ['Sleeps_' + comp for comp in sleeps_df.columns]

        # with open(LOG_FILE, 'a') as lf:
        #     lf.write(str(datetime.datetime.now()) + ' exercise' + '\n')

        exercise_df = self.get_raw_exercises(X)
        exercise_df.columns = ['Exercises_' + comp for comp in exercise_df.columns]

        merged_df = self.dfs['glucose'].join([meals_df, sleeps_df, exercise_df])
        print(merged_df.shape)
        # merged_df = self.dfs['glucose'].join(meals_df)
        # merged_df = merged_df.join(sleeps_5min)
        # merged_df = merged_df.join(exercise_5min)
        merged_df = merged_df.fillna(0)

        assert (merged_df.shape[0] == self.dfs['glucose'].shape[0])

        def tshift(df, lag):  # shifts the temporal values by a given lag
            return df.reset_index(level=['ConnectionID'], drop=True).tshift(freq=lag)

        lags = [str(d) + "min" for d in range(0, HOURS_BACKWARD * 60 + DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN)]
        past_values = X.copy()
        for lag in lags:
            # with open(LOG_FILE, 'a') as lf:
            #     lf.write(str(datetime.datetime.now()) + ' lag: ' + str(lag) + '\n')
            print(str(datetime.datetime.now()) + " - temporal - " + lag)
            temp = merged_df.groupby(level='ConnectionID').apply(tshift, lag=lag)
            temp.columns = [comp + '_before_' + lag for comp in merged_df.columns]
            past_values = past_values.join(temp).ffill(0)

        return [past_values.loc[X.index]]


    def get_raw_glucose(self, X):
        """
        Get the previous glucose measurements 
        :param X: 
        :return: 
        """
        print(str(datetime.datetime.now()) + " - " +
              "[Predictor:get_raw_glucose] start collecting past glucose values...")

        def tshift(df, lag): # shifts the glucose values by a given lag
            return df.reset_index(level=['ConnectionID'], drop=True).tshift(freq=lag)
        
        lags = [str(d) + "min" for d in range(0,HOURS_BACKWARD*60+DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN)]
        past_glucose_values = X.copy()
        for lag in lags:
            print (str(datetime.datetime.now()) + " - glucose - " + lag)
            temp = self.dfs['glucose'].groupby(level='ConnectionID').apply(tshift, lag=lag)
            temp.columns = ['Glucose_before_' + lag]
            past_glucose_values = past_glucose_values.join(temp).ffill()
        
        return [past_glucose_values.loc[X.index]]

    def get_raw_meals(self, X):
        """
        Get the previous meals measurements
        :param X:
        :return:
        """

        print(str(datetime.datetime.now()) + " - " +
              "[Predictor:get_raw_meals] start collecting past meals values...")
        meals_df = self.dfs['meals'].reset_index('MealEventID', drop=True)\
            .set_index('Timestamp', append=True).sort_index()
        meals_df = meals_df[MEALS_LABELS].fillna(0)
        meals_df = meals_df.reset_index(level='Timestamp', drop=False)
        meals_df['Timestamp'] = pd.DatetimeIndex(meals_df['Timestamp']).round(str(DATA_RESOLUTION_MIN) + 'min')
        meals_df.set_index('Timestamp', inplace=True, append=True)
        meals_df = meals_df.fillna(0)
        meals_df = meals_df.groupby(level=['ConnectionID', 'Timestamp']).apply(np.sum)
        # meals_df = self.dfs['glucose'].join(meals_df)
        # meals_df = meals_df.drop('GlucoseValue', 1).fillna(0)

        testfood_df = self.dfs['test_food'].reset_index(level='EventID', drop=True).set_index('Timestamp', append=True).sort_index()
        testfood_df = testfood_df[MEALS_LABELS].fillna(0)
        testfood_df = testfood_df.reset_index(level='Timestamp', drop=False)
        testfood_df['Timestamp'] = pd.DatetimeIndex(testfood_df['Timestamp']).round(str(DATA_RESOLUTION_MIN) + 'min')
        testfood_df.set_index('Timestamp', inplace=True, append=True)
        testfood_df = testfood_df.fillna(0)
        testfood_df = testfood_df.groupby(level=['ConnectionID', 'Timestamp']).apply(np.sum)

        meals_df = meals_df.reset_index(level='Timestamp', drop=False)
        testfood_df = testfood_df.reset_index(level='Timestamp', drop=False)
        meals_df = pd.concat([meals_df, testfood_df], axis=0)
        meals_df.set_index('Timestamp', inplace=True, append=True)
        meals_df = meals_df.sort_index()
        meals_df = meals_df.groupby(level=['ConnectionID', 'Timestamp']).apply(np.sum)

        return meals_df

    def get_raw_sleep(self, X):
        """
        Get the previous sleep measurements
        :param X:
        :return:
        """
        print(str(datetime.datetime.now()) + " - " + "[Predictor:get_raw_sleep] start collecting past sleep values...")


        sleeps_5min = self.dfs['sleep'].copy()
        sleeps_5min = sleeps_5min.fillna(0)
        keep_lines = []

        # with open(LOG_FILE, 'a') as lf:
        #     lf.write(str(datetime.datetime.now()) + " " + str(sleeps_5min.shape) + 'sleep shape' + '\n')

        for i in range(sleeps_5min.shape[0]):
            temp = sleeps_5min.iloc[i]
            if (temp.wakeup_time != 0 and temp.sleep_time != 0 and
                    (pd.Timedelta(pd.to_datetime(temp.wakeup_time) - pd.to_datetime(temp.sleep_time)) <
                         pd.Timedelta('1 days 00:00:00'))):
                keep_lines.append(i)

        sleeps_5min = sleeps_5min.iloc[keep_lines]

        # with open(LOG_FILE, 'a') as lf:
        #     lf.write(str(datetime.datetime.now()) + " " + str(sleeps_5min.shape) + 'sleep shape after parse' + '\n')

        sleeps_5min['sleep_time'] = pd.DatetimeIndex(sleeps_5min['sleep_time']).round(str(DATA_RESOLUTION_MIN) + 'min')
        sleeps_5min['wakeup_time'] = pd.DatetimeIndex(sleeps_5min['wakeup_time']).round(str(DATA_RESOLUTION_MIN) + 'min')
        sleeps_5min = sleeps_5min.reset_index(level='ConnectionID', drop=False)
        sleeps_5min = sleeps_5min.reset_index(level='RunningIndex', drop=False)
        sleeps_5min['sleep_time'] = pd.to_datetime(sleeps_5min['sleep_time'])
        sleeps_5min['wakeup_time'] = pd.to_datetime(sleeps_5min['wakeup_time'])
        # TODO: maybe use this feature as well, problem is it is missing a lot
        sleeps_5min.drop('Quality', axis=1, inplace=True)
        sleeps_5min['ID'] = sleeps_5min.index
        sleeps_5min = pd.melt(sleeps_5min, id_vars=['ID', 'RunningIndex', 'ConnectionID'], var_name=['D'],
                              value_name='Timestamp')
        del sleeps_5min['D']
        sleeps_5min = sleeps_5min.set_index(['ID', 'Timestamp'])
        sample_every_5_min_func = lambda df: df.asfreq(str(DATA_RESOLUTION_MIN) + 'min', method='ffill')
        sleeps_5min.sort_index(level='ID')
        sleeps_5min = sleeps_5min.reset_index(level=0).groupby('ID').apply(sample_every_5_min_func)
        sleeps_5min = sleeps_5min.set_index('ConnectionID', append=True)
        sleeps_5min = sleeps_5min.reset_index(level='ID', drop=True)
        sleeps_5min.drop('RunningIndex', axis=1, inplace=True)
        sleeps_5min.ID = 1
        sleeps_5min.columns = ['slept']
        sleeps_5min = sleeps_5min.reset_index(level=['ConnectionID', 'Timestamp'], drop=False)
        sleeps_5min = sleeps_5min.set_index(['ConnectionID', 'Timestamp'])
        # sleeps_5min = self.dfs['glucose'].join(sleeps_5min)
        # sleeps_5min.drop('GlucoseValue', axis=1, inplace=True)
        # sleeps_5min = sleeps_5min.fillna(0)
        sleeps_5min = sleeps_5min.groupby(level=['ConnectionID', 'Timestamp']).first()
        return sleeps_5min

    def get_raw_exercises(self, X):
        """
        Get the previous exercises measurements
        :param X:
        :return:
        """
        print(str(datetime.datetime.now()) + " - " +
              "[Predictor:get_raw_exercises] start collecting past exercises values...")

        exercise_5min = self.dfs['exercises'].copy()

        def add_t(x): # create a new column which is Timestamp + Duration
            return pd.to_datetime(x.Timestamp) + datetime.timedelta(minutes=x.Duration)

        exercise_5min['exer_start'] = pd.DatetimeIndex(exercise_5min['Timestamp']).round(str(DATA_RESOLUTION_MIN) + 'min')
        exercise_5min['exer_end'] = exercise_5min.apply(add_t, axis=1)
        exercise_5min['exer_end'] = pd.DatetimeIndex(exercise_5min['exer_end']).round(str(DATA_RESOLUTION_MIN) + 'min')
        exercise_5min.drop('Timestamp', axis=1, inplace=True)
        exercise_5min = exercise_5min.reset_index(level='ConnectionID', drop=False)
        exercise_5min = exercise_5min.reset_index(level='EventID', drop=False)
        exercise_5min['exer_start'] = pd.to_datetime(exercise_5min['exer_start'])
        exercise_5min['exer_end'] = pd.to_datetime(exercise_5min['exer_end'])
        # TODO: perhaps we would want to keep this column
        exercise_5min.drop('ExerciseType', axis=1, inplace=True)
        exercise_5min.drop('Duration', axis=1, inplace=True)
        exercise_5min.drop('EventID', axis=1, inplace=True)
        exercise_5min['ID'] = exercise_5min.index
        exercise_5min = pd.melt(exercise_5min, id_vars=['ID', 'Intensity', 'ConnectionID'], var_name=['D'],
                                value_name='Timestamp')
        del exercise_5min['D']
        exercise_5min = exercise_5min.set_index(['ID', 'Timestamp'])
        sample_every_5_min_func = lambda df: df.asfreq(str(DATA_RESOLUTION_MIN) + 'min', method='ffill')
        exercise_5min.sort_index(level='ID')
        exercise_5min = exercise_5min.reset_index(level=0).groupby('ID').apply(sample_every_5_min_func)
        exercise_5min = exercise_5min.set_index('ConnectionID', append=True)
        exercise_5min = exercise_5min.reset_index(level='ID', drop=True)
        exercise_5min.ID = 1 # treat as boolean
        exercise_5min.columns = EXERCISES_LABELS
        exercise_5min = exercise_5min.reset_index(level=['ConnectionID', 'Timestamp'], drop=False)
        exercise_5min = exercise_5min.set_index(['ConnectionID', 'Timestamp'])
        exercise_5min.sort_index(level='ConnectionID')
        # exercise_5min = self.dfs['glucose'].join(exercise_5min) # join with glucose in order to fill non-exercise times
        # exercise_5min.drop('GlucoseValue', axis=1, inplace=True)
        # exercise_5min = exercise_5min.fillna(0) # assign non-exercise time as 0
        # TODO: for some reason there some rows which repeats, should be a bug in the data, so keep only first
        exercise_5min = exercise_5min.groupby(level=['ConnectionID', 'Timestamp']).first()
        return exercise_5min

    def get_glucose_features(self, X):
        """
        Get dataframe with features of glucose levels from the past 12 hours
        :param X: A pandas DataFrame of rows of the form (connectionID, timestamp)
        :return: Dataframe with same indices as X and columns features of glucose levels.
        """
        print(str(datetime.datetime.now()) + " - " +
              "[Predictor:get_glucose_features] start building glucose features...")
        def glucose_lin_fit(glucose):
            x = np.arange(glucose.shape[0])
            y = glucose

            # extreme case - first entry in each person
            if glucose.shape[0] == 1:
                return 1
            return np.polyfit(x, y, 1)[0]

        glucose_features = []

        # mean
        glucose_lags = self.add_rolling_lags(X, self.dfs['glucose'], self.dfs['glucose'].columns,
                                             ['30min', '1h', '4h', '12h'], np.mean, 'mean')
        glucose_features.append(glucose_lags)

        # max
        glucose_max = self.add_rolling_lags(X, self.dfs['glucose'],
                                            self.dfs['glucose'].columns, ['12h'], np.max, 'max')
        glucose_features.append(glucose_max)

        # linear fit
        glucose_fit = self.add_rolling_lags(X, self.dfs['glucose'], self.dfs['glucose'].columns,
                                            ['45min', '1h', '90min'], glucose_lin_fit, 'linear_fit')
        glucose_features.append(glucose_fit)

        return glucose_features

    # TODO: think if these should stay
    def get_meals_features(self, X):
        """
        Get dataframe with features of meals levels from the past 12 hours
        :param X: A pandas DataFrame of rows of the form (connectionID, timestamp)
        :return: Dataframe with same indices as X and columns features of meals levels.
        """
        print(str(datetime.datetime.now()) + " - " + "[Predictor:get_meals_features] start building meals features...")

        meals_features = []

        meals_df = self.dfs['meals'].reset_index('MealEventID', drop=True).set_index('Timestamp', append=True)

        meals_labels = MEALS_LABELS
        meals_df = meals_df[meals_labels].fillna(0)

        # merge(sum) meals from exact same timestamp
        meals_df = meals_df.groupby(level=['ConnectionID', 'Timestamp']).apply(np.sum)

        # sum rolling meals
        meals_lags = self.add_rolling_lags(X, meals_df, meals_labels, ['15min', '30min', '1h'], np.sum, 'sum')
        meals_features.append(meals_lags)

        return meals_features

    def get_blood_features(self, X):
        """
        Get features of blood tests
        :param X: A pandas DataFrame of rows of the form (connectionID, timestamp)
        :return: List of dataframes indexed by "connection ID", and data of blood tests.
        """
        print(str(datetime.datetime.now()) + " - " + "[Predictor:get_blood_features] start building blood features...")
        # get all blood features
        blood_features = self.dfs['blood']
        blood_features = blood_features[BLOOD_LABELS]

        # fill na with median of training data
        if (os.path.exists(HOME_DIR + 'blood_median.pickle')):
            with open(HOME_DIR + 'blood_median.pickle', 'rb') as handle:
                train_blood_features_median = pickle.load(handle)
        else:
            train_blood_features_median = blood_features.median()
            with open(HOME_DIR + 'blood_median.pickle', 'wb') as handle:
                pickle.dump(train_blood_features_median, handle, protocol=pickle.HIGHEST_PROTOCOL)

        blood_features = blood_features.fillna(train_blood_features_median)
        return [blood_features]

    def get_bacterial_features(self, X):
        """
        Get features of bacterial species
        :param X: A pandas DataFrame of rows of the form (connectionID, timestamp)
        :return: List of dataframes indexed by "connection ID", and first PCs.
        """
        print(str(datetime.datetime.now()) + " - " +
              "[Predictor:get_bacterial_features] start building bacterial features...")
        # since values in bacterial abundance matrix are log10 values, we replace NaN with an
        # artificial value of zero which is a smaller negative integer
        artificial_zero = np.floor(self.dfs['bac'].min().min()) - 1
        bac_features = self.dfs['bac']
        bac_features = bac_features.fillna(artificial_zero)

        # In order for the PC space to be consistent, we load the train data and perform PCA over it
        # only then we project the new data over that PC space
        # During training, this is not needed.
        orig_bacteria_df = pd.read_pickle(SRC_DIR + 'data/BacterialSpecies.df')
        artificial_zero_orig = np.floor(orig_bacteria_df.min().min()) - 1
        bac_features_orig = orig_bacteria_df.fillna(artificial_zero_orig)
        bac_features_orig = bac_features_orig.values
        pca = PCA(n_components=5)
        # compute PC space
        pca.fit_transform(bac_features_orig)

        # project current data
        bac_pca = pca.transform(bac_features)
        bac_df = pd.DataFrame(bac_pca)
        bac_df.index = bac_features.index
        bac_df.columns = ["bac_PC" + str(i) for i in bac_df.columns]
        return [bac_df]

    def get_measurements_features(self, X):
        """
        Get features of measurements tests
        :param X: A pandas DataFrame of rows of the form (connectionID, timestamp)
        :return: List of dataframes indexed by "connection ID", and data of measurements.
        """
        print(str(datetime.datetime.now()) + " - " +
              "[Predictor:get_measurements_features] start building measurements features...")
        measurements_features = self.dfs['measurements']
        measurements_features = measurements_features[MEASUREMENTS_LABELS]

        # fill na with median of training data
        if (os.path.exists(HOME_DIR + 'measurements_median.pickle')):
            with open(HOME_DIR + 'measurements_median.pickle', 'rb') as handle:
                train_measurements_features_median = pickle.load(handle)
        else:
            train_measurements_features_median = measurements_features.median()
            with open(HOME_DIR + 'measurements_median.pickle', 'wb') as handle:
                pickle.dump(train_measurements_features_median, handle, protocol=pickle.HIGHEST_PROTOCOL)

        measurements_features = measurements_features.fillna(train_measurements_features_median)
        return [measurements_features]

    def build_features(self, X, train = False):
        """ Enhance the given table of X=(connectionIDs, timestamps) to a have more relevant data.
        Eventually it would be better to have a big table with lots of features for each row(connectionID, timestamp)
        :param X: A pandas DataFrame of rows of the form (connectionID, timestamp)
        :return: Enhanced dataframe with features
        """
        print(str(datetime.datetime.now()) + " - " + "[Predictor:build_features] start building features...")
        personal_f = []
        temporal_f = []

        # collect PERSONAL features:
        personal_f += self.get_blood_features(X)
        personal_f += self.get_measurements_features(X)
        personal_f += self.get_bacterial_features(X)

        # join personal features and adjust their indices:
        personal_data = pd.concat(personal_f, 1)
        personal_data = X.join(personal_data, how='inner')
        personal_data = personal_data.loc[X.index]

        # save a lot of time when we want to build features
        if (train):
            if (os.path.exists(HOME_DIR + 'x_with_features_raw_including_meals_sleep_exercise_' + str(DATA_RESOLUTION_MIN) + 'min' + '.pickle')):
                print(str(datetime.datetime.now()) + " - " + "[Predictor:build_features] raw features were loaded from pickle")
                with open(HOME_DIR + 'x_with_features_raw_including_meals_sleep_exercise_' + str(DATA_RESOLUTION_MIN) + 'min' + '.pickle', 'rb') as handle:
                    raw_temporal_features = pickle.load(handle)
                    # get all raw features (glucose, meals, sleep, exercise)
                    raw_temporal_features = [raw_temporal_features.filter(regex='_before_')]
            else:
                raw_temporal_features = self.get_raw_temporal_features(X)
        else:
            raw_temporal_features = self.get_raw_temporal_features(X)

        temporal_f += raw_temporal_features

        # collect TEMPORAL features
        temporal_f += self.get_glucose_features(X)
        temporal_f += self.get_time_features(X)
        temporal_f += self.get_meals_features(X)

        # concatenate all temporal feature vectors together to form on big enhanced dataframe
        temporal_data = pd.concat(temporal_f, 1)
        # keep only the rows from the original X
        temporal_data = temporal_data.loc[X.index]

        features_data = pd.concat([personal_data, temporal_data], axis=1)

        return features_data

    def add_rolling_lags(self, X, features_df, cols, lags, agg_func, agg_name):
        """ An example of an enhancing rolling function
        :param features_df: the dataframe that would be used for enhancing X
        :param cols: which cols we want to use for the enhancement
        :param lags: which lags to use for the rolling window
        :param agg_func: what aggregation function to use on each window
        :param agg_name: a string to be used for the resulting feature column
        :return: the original X dataframe together with the feature columns
        """
        # concat the features values to X to get a bigger table with all of them together
        df = pd.concat([X, features_df[cols]], axis=1)

        # for each (connectionID,timestamp) and for each lag, aggregate according to agg_func
        features = []

        for lag in lags:
            print(str(datetime.datetime.now()) + " - " + lag)

            def roll_lag(group):
                return group.reset_index(level=['ConnectionID'], drop=True).rolling(lag).apply(agg_func)

            f = df.fillna(0).sort_index().groupby(level='ConnectionID').apply(roll_lag)

            f.columns = ['{}_{}_{}'.format(col, agg_name, lag) for col in f.columns]
            features.append(f)
        # concatenate all feature vectors together to form on big enhanced dataframe
        df = pd.concat(features, 1)
        # keep only the rows from the original X
        df = df.loc[X.index]
        return df

    def train(self, basic_x, y):
        """ Train the network on a data matrix
        """
        # build features for set of (connID, timestamp)
#         x = self.build_features(basic_x, True)
# #
# # #         pickle to save run_time
#         with open(HOME_DIR + 'x_with_features_raw_including_meals_sleep_exercise_' + str(DATA_RESOLUTION_MIN) + 'min' + '.pickle', 'wb') as handle:
#             print(str(datetime.datetime.now()) + " - " + "[Predictor:train] x with features saved to pickle")
#             pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         return

        # # original
        with open(HOME_DIR + 'x_with_features_raw_including_meals_sleep_exercise_' + str(DATA_RESOLUTION_MIN) + 'min' + '.pickle', 'rb') as handle:
            print(str(datetime.datetime.now()) + " - " + "[Predictor:train] x with features was loaded from pickle")
            x = pickle.load(handle)

        # normalized
        # with open('/home/noamba/Genie/Microbiome/Analyses/Noamba/x_with_features_raw_including_meals_sleep_exercise_5min_norm.pickle', 'rb') as handle:
        #     print(str(datetime.datetime.now()) + " - " + "[Predictor:train] x with features was loaded from pickle")
        #     x = pickle.load(handle)


        if (DAY_or_NIGHT == "NIGHT"):
            print ("night")
            ind = np.where(x['time'] < 6)
            y = y[ind]
            x = x.loc[x['time'] < 6]
        elif (DAY_or_NIGHT == "DAY"):
            print("day")
            ind = np.where(x['time'] >= 6)
            y = y[ind]
            x = x.loc[x['time'] >= 6]
        elif (DAY_or_NIGHT == ''):
            pass

        # initialize network object:
        features_size = x.shape[1]
        # print(features_size)
        net_pred = GlucoseNetwork(features_size, model_path=self.model_path)

        # train the network parameters
        net_pred.train_network(x, y, train_part=TRAIN_PART, epochs=EPOCHS, batch_size=BATCH_SIZE)


    # def predict(self, X):#DO NOT CHANGE THIS SIGNATURE
    #     """ Given dataFrame of connection id and time stamp (X) predict
    #     the glucose level of each connection id 2 hours after timestamp
    #
    #     :param X: Empty Multiindex dataframe with indexes ConnectionID and Timestamp
    #     :return: numpy array with the predictions for each row in X
    #              (which is the following number for each row: glucose[timestamp+1hour] - glucose[timestamp])
    #     """
    #
    #     with open(LOG_FILE, 'a') as lf:
    #         lf.write(str(datetime.datetime.now()) + ' start building features' + '\n')
    #
    #     # build features for set of (connID, timestamp)
    #     x = self.build_features(X, False)
    #
    #     with open(LOG_FILE, 'a') as lf:
    #         lf.write(str(datetime.datetime.now()) + ' finish building features' + '\n')
    #
    #     # with open(HOME_DIR + 'x_with_features_raw_including_meals_sleep_exercise_' + str(DATA_RESOLUTION_MIN) + 'min' + '.pickle', 'rb') as handle:
    #     #     print(str(datetime.datetime.now()) + " - " + "[Predictor:predict] x with features was loaded from pickle")
    #     #     x = pickle.load(handle)
    #
    #
    #     # TODO insert this to a static function of the network class?
    #     features_size = x.shape[1]
    #     models_to_use = MODELS_TO_USE
    #
    #     x_raw = x.filter(regex='_before_')
    #     temp_l = [x_raw.filter(regex='Meals_' + comp + '_before') for comp in MEALS_LABELS]
    #     temp_l += [x_raw.filter(regex='Exercises_' + comp + '_before') for comp in EXERCISES_LABELS]
    #     temp_l.append(x_raw.filter(regex='Sleeps_slept_before'))
    #     temp_l.append(x_raw.filter(regex='GlucoseValue_before'))
    #
    #     x_raw = np.dstack([com.values for com in temp_l])
    #     x_other = x.select(lambda x: not re.search('_before_', x), axis=1).values
    #
    #     with open(HOME_DIR + 'linear_regression_coef.pickle', 'rb') as handle:
    #         lr_coef = pickle.load(handle)
    #     if (str(models_to_use) in lr_coef):
    #         lr_coef = lr_coef[str(models_to_use)]
    #     else:
    #         lr_coef = [1./len(models_to_use) for i in range(len(models_to_use))]
    #     i = 0
    #
    #     predictions = []
    #     for model in models_to_use:
    #         # with open(LOG_FILE, 'a') as lf:
    #         #     lf.write(str(datetime.datetime.now()) + ' - ' + model + ' start predicting' + '\n')
    #         net_pred = GlucoseNetwork(features_size, model_path=model)
    #         y = net_pred.predict_network([x_raw, x_other])
    #         y = y.ravel()
    #         predictions.append(y * lr_coef[i])
    #         i += 1
    #         with open(LOG_FILE, 'a') as lf:
    #             lf.write(str(datetime.datetime.now()) + ' - ' + model + ' finish predicting' + '\n')
    #
    #     return sum(predictions)

    def predict(self, X):#DO NOT CHANGE THIS SIGNATURE
        """ Given dataFrame of connection id and time stamp (X) predict
        the glucose level of each connection id 2 hours after timestamp

        :param X: Empty Multiindex dataframe with indexes ConnectionID and Timestamp
        :return: numpy array with the predictions for each row in X
                 (which is the following number for each row: glucose[timestamp+1hour] - glucose[timestamp])
        """

        with open(LOG_FILE, 'a') as lf:
            lf.write(str(datetime.datetime.now()) + ' start building features' + '\n')

        # build features for set of (connID, timestamp)
        x = self.build_features(X, False)

        if (NORM == 'NORM'):
            labels_to_norm = x.filter(regex='_before_0min').columns
            labels_to_norm = [x.split('0min')[0] for x in labels_to_norm]
            labels_to_norm = [x for x in labels_to_norm if x[0] == 'G' or x[0] == 'M']
            temp_means = [x.filter(regex=h).values.mean() for h in labels_to_norm]
            temp_std = [x.filter(regex=h).values.std() for h in labels_to_norm]
            cols = [x.filter(regex=l).columns for l in labels_to_norm]
            for c in range(len(cols)):
                x[cols[c]] = (x[cols[c]] - temp_means[c]) / temp_std[c]
                print(c)

        with open(LOG_FILE, 'a') as lf:
            lf.write(str(datetime.datetime.now()) + ' finish building features' + '\n')

        # with open(HOME_DIR + 'x_with_features_raw_including_meals_sleep_exercise_' + str(DATA_RESOLUTION_MIN) + 'min' + '.pickle', 'rb') as handle:
        #     print(str(datetime.datetime.now()) + " - " + "[Predictor:predict] x with features was loaded from pickle")
        #     x = pickle.load(handle)
        night_ind = np.where(x['time'] < 6)

        # TODO insert this to a static function of the network class?
        features_size = x.shape[1]
        models_to_use = MODELS_TO_USE
        night_models = NIGHT_MODELS



        x_raw = x.filter(regex='_before_')
        temp_l = [x_raw.filter(regex='Meals_' + comp + '_before') for comp in MEALS_LABELS]
        temp_l += [x_raw.filter(regex='Exercises_' + comp + '_before') for comp in EXERCISES_LABELS]
        temp_l.append(x_raw.filter(regex='Sleeps_slept_before'))
        temp_l.append(x_raw.filter(regex='GlucoseValue_before'))

        x_raw = np.dstack([com.values for com in temp_l])
        x_other = x.select(lambda x: not re.search('_before_', x), axis=1).values


        predictions = []
        for model in models_to_use:
            # with open(LOG_FILE, 'a') as lf:
            #     lf.write(str(datetime.datetime.now()) + ' - ' + model + ' start predicting' + '\n')
            net_pred = GlucoseNetwork(features_size, model_path=model)
            y = net_pred.predict_network([x_raw, x_other])
            y = y.ravel()
            predictions.append(y / len(models_to_use))
            with open(LOG_FILE, 'a') as lf:
                lf.write(str(datetime.datetime.now()) + ' - ' + model + ' finish predicting' + '\n')

        night_predictions = []
        for model in night_models:
            # with open(LOG_FILE, 'a') as lf:
            #     lf.write(str(datetime.datetime.now()) + ' - ' + model + ' start predicting' + '\n')
            net_pred = GlucoseNetwork(features_size, model_path=model)
            y = net_pred.predict_network([x_raw, x_other])
            y = y.ravel()
            night_predictions.append(y / len(night_models))
            with open(LOG_FILE, 'a') as lf:
                lf.write(str(datetime.datetime.now()) + ' - ' + model + ' finish predicting' + '\n')

        if (len(night_predictions) > 0):
            y_night = sum(night_predictions)
            y = sum(predictions)
            y[night_ind] = y[night_ind]/2
            y[night_ind] += y_night[night_ind]/2

        return y


    def compute_linear_regression(self, Y):
        with open(HOME_DIR + 'x_with_features_raw_including_meals_sleep_exercise_' + str(DATA_RESOLUTION_MIN) + 'min' + '.pickle', 'rb') as handle:
            print(str(datetime.datetime.now()) + " - " + "[Predictor:predict] x with features was loaded from pickle")
            x = pickle.load(handle)
        features_size = x.shape[1]
        models_to_use = MODELS_TO_USE

        x_raw = x.filter(regex='_before_')
        temp_l = [x_raw.filter(regex='Meals_' + comp + '_before') for comp in MEALS_LABELS]
        temp_l += [x_raw.filter(regex='Exercises_' + comp + '_before') for comp in EXERCISES_LABELS]
        temp_l.append(x_raw.filter(regex='Sleeps_slept_before'))
        temp_l.append(x_raw.filter(regex='GlucoseValue_before'))

        x_raw = np.dstack([com.values for com in temp_l])
        x_other = x.select(lambda x: not re.search('_before_', x), axis=1).values

        predictions = []
        for model in models_to_use:
            net_pred = GlucoseNetwork(features_size, model_path=model)
            y = net_pred.predict_network([x_raw, x_other])
            y = y.ravel()
            predictions.append(y)
        predictions_X = np.array(predictions)
        predictions_X = predictions_X.transpose()

        predictions_X.shape = (predictions_X.shape[0], predictions_X.shape[1])

        regr = linear_model.LinearRegression()
        regr.fit(predictions_X, Y)

        with open(HOME_DIR + 'linear_regression_coef.pickle', 'rb') as handle:
            coef = pickle.load(handle)

        coef[str(MODELS_TO_USE)] = regr.coef_

        with open(HOME_DIR + 'linear_regression_coef.pickle', 'wb') as handle:
            pickle.dump(coef, handle, protocol=pickle.HIGHEST_PROTOCOL)

        y_pred = Y*0
        for i in range(len(models_to_use)):
            y_pred += regr.coef_[i] * predictions[i]

        score = 0
        for i in range(len(Y)):
            score += np.square(Y[i]-y_pred[i])
        score = score/len(Y)
        print(str(datetime.datetime.now()) + " - " + "Your score (MSE) is: {}".format(score))

        print(str(datetime.datetime.now()) + " - " + "cor = ", pearsonr(Y, y_pred))
        return


    # computes the mean sequared loss
    def compute_loss(self, y_pred, Y):
        score = 0
        for i in range(len(Y)):
            score += np.square(Y[i]-y_pred[i])
        return score/len(Y)

    # test features for HW5
    #TODO: update so it would work with sleep, exercise and meals raw features
    def feature_test(self, Y):

        with open(HOME_DIR + 'x_with_features_raw.pickle', 'rb') as handle:
            print(str(datetime.datetime.now()) + " - " + "[Predictor:train] x with features was loaded from pickle")
            x = pickle.load(handle)

        features_size = x.shape[1]
        net_pred = GlucoseNetwork(features_size, model_path=self.model_path)
        x_raw = x.filter(regex='GlucoseValue_before_')
        x_other = x.select(lambda x: not re.search('GlucoseValue_before_', x), axis=1)

        blood = BLOOD_LABELS
        measurements = MEASUREMENTS_LABELS
        bac_pca = ['bac_PC0', 'bac_PC1', 'bac_PC2', 'bac_PC3', 'bac_PC4']
        glucose_mean = ['GlucoseValue_mean_30min', 'GlucoseValue_mean_1h', 'GlucoseValue_mean_4h',
                        'GlucoseValue_mean_12h']
        glucose_max = ['GlucoseValue_max_12h']
        glucose_fit = ['GlucoseValue_linear_fit_45min', 'GlucoseValue_linear_fit_1h',
                       'GlucoseValue_linear_fit_90min']
        time = ['time']
        food = ['Weight_sum_15min', 'Protein_g_sum_15min', 'TotalLipid_g_sum_15min',
                'Carbohydrate_g_sum_15min', 'Water_g_sum_15min', 'Alcohol_g_sum_15min',
                'Energy_kcal_sum_15min', 'Weight_sum_30min', 'Protein_g_sum_30min',
                'TotalLipid_g_sum_30min', 'Carbohydrate_g_sum_30min', 'Water_g_sum_30min',
                'Alcohol_g_sum_30min', 'Energy_kcal_sum_30min', 'Weight_sum_1h', 'Protein_g_sum_1h',
                'TotalLipid_g_sum_1h', 'Carbohydrate_g_sum_1h', 'Water_g_sum_1h', 'Alcohol_g_sum_1h',
                'Energy_kcal_sum_1h']

        predictions = []
        names = ['blood', 'measurements', 'bac_pca', 'glucose_mean', 'glucose_max', 'glucose_fit', 'time', 'food',
                 'raw glucose', 'all']

        groups = [blood, measurements, bac_pca, glucose_mean, glucose_max, glucose_fit, time, food]
        x_raw_0 = x_raw.values * 0.0
        # set all other to 0
        for group, name in zip(groups, names):
            print(name)
            x_other_tmp = x_other.copy()
            x_other_tmp[[feature for feature in x_other.columns.values if feature not in group]] = 0
            y = net_pred.predict_network([x_raw_0, x_other_tmp.values])
            y = y.ravel()
            predictions.append(y)
        y = net_pred.predict_network([x_raw.values, x_other.values * 0])
        y = y.ravel()
        predictions.append(y)

        y = net_pred.predict_network([x_raw.values, x_other.values])
        y = y.ravel()
        predictions.append(y)
        # set group to 0
        for group, name in zip(groups, names):
            print(name)
            x_other_tmp = x_other.copy()
            x_other_tmp[[feature for feature in x_other.columns.values if feature in group]] = 0
            y = net_pred.predict_network([x_raw.values, x_other_tmp.values])
            y = y.ravel()
            predictions.append(y)
        y = net_pred.predict_network([x_raw.values * 0, x_other.values])
        y = y.ravel()
        predictions.append(y)

        y = net_pred.predict_network([x_raw.values * 0, x_other.values * 0])
        y = y.ravel()
        predictions.append(y)

        losses = [self.compute_loss(y_pred, Y) for y_pred in predictions]
        corrs = [pearsonr(y_pred, Y) for y_pred in predictions]

        with open(HOME_DIR + 'feature_test_full.pickle', 'wb') as handle:
            pickle.dump([names, losses, corrs, predictions], handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path2data', help='Path to data directory', type=str, default='./data')
    parser.add_argument('-t', '--train', type=bool, help='whether to train the network', default=False)
    parser.add_argument('-p', '--predict', type=bool, help='whether to predict', default=False)
    parser.add_argument('-f', '--features_test', type=bool, help='whether to test features', default=False)
    parser.add_argument('-lr', '--linear_regression', type=bool, help='whether to perform linear regression', default=False)
    command_args = parser.parse_args()

    if not (command_args.train or command_args.predict or command_args.features_test or command_args.linear_regression):
        print(str(datetime.datetime.now()) + " - " + "[main] Everything is False! exiting...")
        return

    # create Predictor instance
    path2data = command_args.path2data
    predict_inst = Predictor(path2data)

    # load x_y table : (connId, timestamp)---> label
    x_y = pd.read_pickle(os.path.join(path2data, 'x_y_raw.df'))
    # x_y = pd.read_pickle(os.path.join(path2data, 'x_y.df'))

    # split the table to X (connId, timestamp) and Y (label)
    X = x_y.drop('label', axis=1)
    Y = np.asarray(x_y['label'].tolist())

    #     train network
    if command_args.train:
        print(str(datetime.datetime.now()) + " - " + "[main] training network...")
        predict_inst.train(X, Y)

    if command_args.predict:
        print(str(datetime.datetime.now()) + " - " + "[main] predicting...")
        tf.reset_default_graph()

        # predict Y
        y_pred = predict_inst.predict(X)


        # test the prediction
        score = 0
        for i in range(len(Y)):
            score += np.square(Y[i]-y_pred[i])
        score = score/len(Y)
        print(str(datetime.datetime.now()) + " - " + "Your score (MSE) is: {}".format(score))

        print(str(datetime.datetime.now()) + " - " + "cor = ", pearsonr(Y, y_pred))

    if command_args.features_test:
        predict_inst.feature_test(Y)

    if command_args.linear_regression:
        predict_inst.compute_linear_regression(Y)

    print(str(datetime.datetime.now()) + " - " + "[main] end of main")

if __name__ == "__main__":
    main()
