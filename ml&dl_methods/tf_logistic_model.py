import sys

sys.path.append('..')
import matplotlib as mpl
import config
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.dates as dt

mpl.use('TkAgg')


def import_csv(stock_code, period_type):
    # period_type:day or min
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
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df['Date'] = df['Date'].apply(lambda x: dt.date2num(x))
    # df.set_index(['Date'], inplace=True)
    return df

symbol = 'sh600519'
period = 'day'

stock_df = import_csv(symbol, period)[-2000:]

# 逻辑回归
NUM_STEPS = 20
wb_ = []
# def sigmoid(x):
#     return 1/(1+np.exp(-x))

x_data = stock_df['Date'].values
y_data = stock_df['Close'].values
# print(type(x_data))

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[1,2000])
    y_true = tf.placeholder(tf.float32, shape=None)

    with tf.name_scope('inference'):
        w = tf.Variable(tf.ones([1, 2000]), dtype=tf.float32, name='weights')
        b = tf.Variable(0, dtype=tf.float32, name='bias')
        y_pred = tf.matmul(w, tf.transpose(x)) + b

    with tf.name_scope('loss'):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        loss = tf.reduce_mean(loss)

    with tf.name_scope('train'):
        learning_rate = .5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train, feed_dict={x: [x_data], y_true: [y_data]})
            print(step, sess.run([w, b]))
            wb_.append(sess.run(tf.matmul(w, tf.transpose(x)) + b))


plt.plot(x_data, wb_, label='y_pred', linewidth=1.0, linestyle='--')
plt.plot(x_data, y_data, label='y_true', linewidth=1.0)
plt.legend(loc='upper right')
plt.show()
