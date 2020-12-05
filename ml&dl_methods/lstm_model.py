import sys, config
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import pandas as pd
import numpy as np
mpl.use('TkAgg')
sys.path.append('..')

def import_csv(stock_code, period_type):
    # perviod_type:day or min
    if period_type == 'day':
        df = pd.read_csv(config.input_data_path + '\\stock_data\\' + stock_code + '.csv')
        df.rename(
            columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'change': 'Change',
                     'volume': 'Volume'}, inplace=True)
    else:
        df = pd.read_csv(config.input_data_path + '\\stock_minute_data\\' + stock_code + '.csv')
        df.rename(columns={'candle_end_time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                           'volume': 'Volume'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(['Date'], inplace=True)
    return df

df = import_csv('sh600520', 'day')

date = tf.placeholder('float', name)