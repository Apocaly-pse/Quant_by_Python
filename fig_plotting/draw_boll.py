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

df = import_csv(symbol, period)[-100:]

# SMA:简单移动平均(Simple Moving Average)
time_period = 20  # SMA的计算周期，默认为20
stdev_factor = 2  # 上下频带的标准偏差比例因子
history = []  # 每个计算周期所需的价格数据
sma_values = []  # 初始化SMA值
upper_band = []  # 初始化阻力线价格
lower_band = []  # 初始化支撑线价格

# 构造列表形式的绘图数据
for close_price in df['Close']:
    #
    history.append(close_price)

    # 计算移动平均时先确保时间周期不大于20
    if len(history) > time_period:
        del (history[0])

    # 将计算的SMA值存入列表
    sma = np.mean(history)
    sma_values.append(sma)
    # 计算标准差
    stdev = np.sqrt(np.sum((history - sma) ** 2) / len(history))
    upper_band.append(sma + stdev_factor * stdev)
    lower_band.append(sma - stdev_factor * stdev)

# 将计算的数据合并到DataFrame
df = df.assign(收盘价=pd.Series(df['Close'], index=df.index))
df = df.assign(middle_line=pd.Series(sma_values, index=df.index))
df = df.assign(resistance_line=pd.Series(upper_band, index=df.index))
df = df.assign(support_line=pd.Series(lower_band, index=df.index))

# 绘图
ax = plt.figure()
# 设定y轴标签
ax.ylabel = '%s price in ￥' % (symbol)

df['收盘价'].plot(color='k', lw=1., legend=True)
df['middle_line'].plot(color='b', lw=1., legend=True)
df['resistance_line'].plot(color='r', lw=1., legend=True)
df['support_line'].plot(color='g', lw=1., legend=True)
plt.show()
