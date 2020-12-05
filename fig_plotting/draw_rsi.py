import sys

sys.path.append('..')
import config, os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')
mpl.rcParams['font.sans-serif'] = 'Microsoft YaHei'

def import_csv(stock_code, period_type):
    # period_type:day or min
    if period_type == 'day':
        df = pd.read_csv(os.path.join(config.input_data_path, 'stock_data', stock_code + '.csv'))
        df.rename(
            columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'change': 'Change',
                     'volume': 'Volume'}, inplace=True)
    else:
        df = pd.read_csv(os.path.join(config.input_data_path, 'stock_minute_data', stock_code + '.csv'))
        df.rename(columns={'candle_end_time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                           'volume': 'Volume'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(['Date'], inplace=True)
    return df


def min1_convert(minute1_df, period):
    convert_df = minute1_df.resample(rule='%dmin' % (period)).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    convert_df.dropna(inplace=True)
    return convert_df


def daily_convert(daily_df, period_type):
    convert_df = daily_df.resample(rule=period_type).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'money': 'sum'})
    convert_df.dropna(inplace=True)
    return convert_df


symbol = 'sh600519'
period = 'day'

df = import_csv(symbol, period)[-500:]




time_period = 20 # 损益的回溯周期
gain_history = [] # 回溯期内的收益历史（无收益为0，有收益则为收益的幅度）
loss_history = [] # 回溯期内的损失历史（无损失为0，有损失则为损失的幅度）
avg_gain_values = [] # 存储平均收益值以便图形绘制
avg_loss_values = [] # 存储平均损失值以便图形绘制
rsi_values = [] # 存储算得的RSI值
last_price = 0
# 当前价 - 过去价 > 0 => 收益(gain).
# 当前价 - 过去价 < 0 => 损失(loss).

# 遍历收盘价以计算 RSI指标
for close in df['Close']:
    if last_price == 0:
        last_price = close

    gain_history.append(max(0, close - last_price))
    loss_history.append(max(0, last_price - close))
    last_price = close

    if len(gain_history) > time_period: # 最大观测值等于回溯周期
        del (gain_history[0])
        del (loss_history[0])

    avg_gain = np.mean(gain_history) # 回溯期的平均收益
    avg_loss = np.mean(loss_history) # 回溯期的平均损失

    avg_gain_values.append(avg_gain)
    avg_loss_values.append(avg_loss)

    # 初始化rs值
    rs = 0
    if avg_loss > 0: # 避免除数为 0，出现错误
        rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))
    rsi_values.append(rsi)

# 将计算所得值并入DataFrame
df = df.assign(RSAvgGainOver20D=pd.Series(avg_gain_values, index=df.index))
df = df.assign(RSAvgLossOver20D=pd.Series(avg_loss_values, index=df.index))
df = df.assign(RSIOver20D=pd.Series(rsi_values, index=df.index))


# 定义画布并添加子图
fig = plt.figure()
ax1 = fig.add_subplot(311, ylabel='%s price in ￥'%(symbol))
df['Close'].plot(ax=ax1, color='black', lw=1., legend=True)

# sharex:设置同步缩放横轴，便于缩放查看
ax2 = fig.add_subplot(312, ylabel='RS', sharex=ax1)
df['RSAvgGainOver20D'].plot(ax=ax2, color='g', lw=1., legend=True)
df['RSAvgLossOver20D'].plot(ax=ax2, color='r', lw=1., legend=True)

ax3 = fig.add_subplot(313, ylabel='RSI', sharex=ax1)
df['RSIOver20D'].plot(ax=ax3, color='b', lw=1., legend=True)
plt.show()



