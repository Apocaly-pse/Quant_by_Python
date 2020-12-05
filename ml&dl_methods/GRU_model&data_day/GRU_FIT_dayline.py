import os,sys
# sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir)))
sys.path.append('..\\..')
import config
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

mpl.use('TkAgg')


def import_csv(stock_code, rows):
    df = pd.read_csv(config.input_data_path + '\\stock_data\\' + stock_code + '.csv')[-rows:]
    df.reset_index(drop=True, inplace=True)
    df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'change']]
    df['date'] = pd.to_datetime(df['date'])
    return df


stock_code = 'sh600519'
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
split_date = datetime(year=2019, month=7, day=1)
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


time_steps = 5
# 生成训练数据
X_train, y_train = makeXy(df_train['scaled_close'], time_steps)
# print(X_train.shape, y_train.shape)
X_val, y_val = makeXy(df_val['scaled_close'], time_steps)
# print(X_val.shape, y_val.shape)
# exit()
X_train, X_val = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

def train_model():
    from keras.layers import Dense, Input, Dropout
    from keras.layers.recurrent import GRU
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint
    from keras.optimizers import Adam

    # 定义GRU神经网络层结构
    input_layer = Input(shape=(time_steps, 1), dtype='float32')
    gru_layer1 = GRU(64, input_shape=(time_steps, 1), return_sequences=True)(input_layer)
    gru_layer2 = GRU(32, input_shape=(time_steps, 64), return_sequences=False)(gru_layer1)
    # gru_layer3 = GRU(32, input_shape=(time_steps, 64), return_sequences=False)(gru_layer2)

    dropout_layer = Dropout(.2)(gru_layer2)

    output_layer = Dense(1, activation='tanh')(dropout_layer)
    optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-7, decay=0.0, amsgrad=False)

    ts_model = Model(inputs=input_layer, outputs=output_layer)
    # 编译模型
    ts_model.compile(loss='mean_absolute_error', optimizer=optimizer)

    # ts_model.summary() # 输出模型层结构

    # 存储损失函数最小值时的模型为hdf文件
    save_best = ModelCheckpoint('%s_GRU_weights.{epoch:02d}-{val_loss:.4f}.h5'%stock_code,
                                monitor='val_loss', verbose=1, save_best_only=True,
                                save_weights_only=False, mode='min', period=1)
    # 开始训练模型(拟合)
    ts_model.fit(x=X_train, y=y_train, batch_size=16,
                 epochs=35, verbose=2, callbacks=[save_best],
                 validation_data=(X_val, y_val), shuffle=True)

# train_model(); exit()

from keras.models import load_model
# 读取最优模型
best_model = load_model('%s_GRU_weights.16-0.0391.h5'%stock_code)
preds = best_model.predict(X_val)
pred_close = np.squeeze(scaler.inverse_transform(preds))


mse = MSE(df_val['close'].loc[time_steps:], pred_close)
print('MSE : ', round(mse, 4))

mae = MAE(df_val['close'].loc[time_steps:], pred_close)
print('MAE : ', round(mae, 4))

plot_steps = 150
plt.figure()
plt.plot(range(plot_steps), df_val['close'].loc[time_steps:plot_steps+time_steps-1], linewidth=.5, color='r')
plt.plot(range(plot_steps), pred_close[:plot_steps], linewidth=.5, color='b')
plt.legend(['Actual', 'Predicted'], loc=2)
plt.title('A_Stock %s close_price'%stock_code)
plt.ylabel('close_price')
plt.xlabel('Index')
# plt.show()
plt.savefig('GRU_%s.jpg'%stock_code, format='jpg', dpi=300)