from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE
from keras.layers import Dense, Input, Dropout
# from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

# import tensorflow as tf
mpl.use('TkAgg')


def import_csv(stock_code, rows):
    df = pd.read_csv(stock_code + '.csv')[-rows:]
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={'candle_end_time': 'date'}, inplace=True)
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df['date'] = pd.to_datetime(df['date'])
    return df


stock_code = 'sh600200'
df = import_csv(stock_code, 2800)
# # draw Box Plot : visualize the central tendency and dispersion of close_price
# plt.figure()
# fig1 = sns.boxplot(df['close'])
# fig1.set_title('Box plot of %s' % stock_code)
# plt.show()
# exit()

# plt.figure()
# fig2 = sns.lineplot(df['date'], df['close'])
# fig2.set_title('Time series of %s'%stock_code)
# plt.show()

# 数据正规化处理
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_close'] = scaler.fit_transform(np.array(df['close']).reshape(-1, 1))
# print(df['scaled_close'])

# 数据集划分处理
split_date = datetime(year=2020, month=4, day=17, hour=9, minute=30,)
df_train = df.loc[df['date'] < split_date]
df_val = df.loc[df['date'] >= split_date]
df_val.reset_index(drop=True, inplace=True)


# print(df_train.shape, df_val.shape)
# exit()
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


# 生成训练数据
X_train, y_train = makeXy(df_train['scaled_close'], 7)
# print(X_train.shape, y_train.shape)
X_val, y_val = makeXy(df_val['scaled_close'], 7)
# print(X_val.shape, y_val.shape)
# exit()

# # 定义神经网络层结构
# input_layer = Input(shape=(7,), dtype='float32')
# dense1 = Dense(32, activation='tanh')(input_layer)
# dense2 = Dense(16, activation='tanh')(dense1)
# dense3 = Dense(16, activation='tanh')(dense2)
#
# dropout_layer = Dropout(.2)(dense3)
#
# output_layer = Dense(1, activation='linear')(dropout_layer)
#
# ts_model = Model(inputs=input_layer, outputs=output_layer)
# # 编译模型
# ts_model.compile(loss='mean_absolute_error', optimizer='adam')
# # ts_model.summary() # 输出模型层结构
#
# # 存储拟合最优的模型为hdf文件
# save_best = ModelCheckpoint('%s_MLP_weights.{epoch:02d}-{val_loss:.4f}.h5'%stock_code,
#                             monitor='val_loss', verbose=0, save_best_only=True,
#                             save_weights_only=False, mode='min', period=1)
# # 开始训练模型(拟合)并存储
# ts_model.fit(x=X_train, y=y_train, batch_size=16,
#              epochs=50, verbose=1, callbacks=[save_best],
#              validation_data=(X_val, y_val), shuffle=True)
# exit()
# 读取最优模型
best_model = load_model('%s_MLP_weights.49-0.0071.h5' % stock_code)
preds = best_model.predict(X_val)
pred_close = np.squeeze(scaler.inverse_transform(preds))

mse = MAE(df_val['close'].loc[7:], pred_close)
print(round(mse, 4))

plt.figure()
plt.plot(range(100), df_val['close'].loc[7:106], linestyle='-', marker='*', color='r')
plt.plot(range(100), pred_close[:100], linestyle='-', marker='.', color='b')
plt.legend(['Actual', 'Predicted'], loc=2)
plt.title('Actual VS Predicted %s close_price' % stock_code)
plt.ylabel('close_price')
plt.xlabel('Index')
# plt.show()
plt.savefig('MLP_%s.jpg' % stock_code, format='jpg', dpi=1000)
