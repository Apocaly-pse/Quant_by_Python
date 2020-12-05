import os,sys
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir)))

import config
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE

mpl.use('TkAgg')


def import_csv(stock_code, rows):
    df = pd.read_csv(os.path.join(config.input_data_path, 'stock_data', stock_code + '.csv'))[-rows:]
    df.reset_index(drop=True, inplace=True)
    df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'change']]
    df['date'] = pd.to_datetime(df['date'])
    return df


stock_code = 'sz000001'
df = import_csv(stock_code, 3500)
# draw Box Plot : visualize the central tendency and dispersion of close_price
# plt.figure()
# fig1 = sns.boxplot(df['close'])
# fig1.set_title('Box plot of %s'%stock_code)
# plt.show()

# plt.figure()
# fig2 = sns.lineplot(df['date'], df['close'])
# fig2.set_title('Time series of %s'%stock_code)
# plt.show()

# 数据正规化处理
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_close'] = scaler.fit_transform(np.array(df['close']).reshape(-1, 1))
# print(df['scaled_close'])

# 数据集划分处理
split_date = datetime(year=2019, month=1, day=21)
df_train = df.loc[df['date'] < split_date]
df_val = df.loc[df['date'] >= split_date]
df_val.reset_index(drop=True, inplace=True)


# print(df_train.shape, df_val.shape)

def makeXy(df, time_steps):
    # 本函数用于生成训练模型的数组数据
    # 使用过去time_steps长度的数据来预测下一天的数据
    X = []
    y = []
    for i in range(time_steps, df.shape[0]):
        X.append(list(df.loc[i - time_steps:i - 1]))
        y.append(df.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y

time_steps = 7
# 生成训练数据
X_train, y_train = makeXy(df_train['scaled_close'], time_steps)
# print(X_train.shape, y_train.shape)
X_val, y_val = makeXy(df_val['scaled_close'], time_steps)
# print(X_val.shape, y_val.shape)

def train_model():
    from keras.models import Input, Model
    from keras.layers import Dense, Dropout
    from keras.callbacks import ModelCheckpoint

    # 定义神经网络层结构
    input_layer = Input(shape=(time_steps,), dtype='float32')
    dense1 = Dense(32, activation='tanh')(input_layer)
    dense2 = Dense(16, activation='tanh')(dense1)
    dense3 = Dense(32, activation='tanh')(dense2)
    dense4 = Dense(16, activation='tanh')(dense3)

    dropout_layer = Dropout(.2)(dense4)

    output_layer = Dense(1, activation='linear')(dropout_layer)

    ts_model = Model(inputs=input_layer, outputs=output_layer)
    # 编译模型
    ts_model.compile(loss='mean_absolute_error', optimizer='adam')
    # ts_model.summary() # 输出模型层结构

    # 存储拟合最优的模型为hdf文件
    save_best = ModelCheckpoint('%s_MLP_weights.{epoch:02d}-{val_loss:.4f}.h5'%stock_code,
                                monitor='val_loss', verbose=0, save_best_only=True,
                                save_weights_only=False, mode='min', period=1)
    # 开始训练模型(拟合)
    ts_model.fit(x=X_train, y=y_train, batch_size=16,
                 epochs=50, verbose=2, callbacks=[save_best],
                 validation_data=(X_val, y_val), shuffle=True)

# train_model(); exit()
# 读取最优模型
from keras.models import load_model
best_model = load_model('%s_MLP_weights.42-0.0051.h5'%stock_code)
preds = best_model.predict(X_val)
pred_close = np.squeeze(scaler.inverse_transform(preds))

mse = MAE(df_val['close'].loc[time_steps:], pred_close)
print(round(mse, 4))

plot_steps = 150
plt.figure()
plt.plot(range(plot_steps), df_val['close'].loc[time_steps:plot_steps+time_steps-1], linewidth=.5, linestyle='-', marker='', color='r')
plt.plot(range(plot_steps), pred_close[:plot_steps], linewidth=.5, linestyle='-', marker='', color='b')
plt.legend(['Actual', 'Predicted'], loc=2)
plt.title('Actual & Predicted %s close_price'%stock_code)
plt.ylabel('close_price')
plt.xlabel('Index')
# plt.show()
plt.savefig('MLP_%s_.jpg'%stock_code, format='png', dpi=300)