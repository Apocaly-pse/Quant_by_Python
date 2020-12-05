import sys

sys.path.append('..')
import config, os
import pandas as pd
import pylab as pl
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')
# plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'


'''
The Moving Average Convergence Divergence
 (MACD) was developed by Gerald Appel, and is based on the differences
 between two moving averages of different lengths, a Fast and a Slow moving
 average. A second line, called the Signal line is plotted as a moving
 average of the MACD. A third line, called the MACD Histogram is
 optionally plotted as a histogram of the difference between the
 MACD and the Signal Line.

 MACD = FastMA - SlowMA

Where:

FastMA is the shorter moving average and SlowMA is the longer moving average.
SignalLine = MovAvg (MACD)
MACD Histogram = MACD - SignalLine
 '''


# 导入数据并做处理
def import_csv(stock_code, period_type):
    # period_type:day or min
    if period_type == 'day':
        df = pd.read_csv(os.path.join(config.input_data_path, 'stock_data', stock_code + '.csv'))
        df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        },
            inplace=True)
    else:
        df = pd.read_csv(os.path.join(config.input_data_path, 'stock_minute_data', stock_code + '.csv'))
        df.rename(columns={
            'candle_end_time': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        },
            inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(['Date'], inplace=True)
    return df


stock_code = 'sz000002'
period_type = 'day'
# 要绘制的图像tick数
tick = 200
df = import_csv(stock_code, period_type)[-tick:]

num_periods_fast = 12  # fast EMA time period
K_fast = 2 / (num_periods_fast + 1)  # fast EMA smoothing factor
ema_fast = 0
num_periods_slow = 26  # slow EMA time period
K_slow = 2 / (num_periods_slow + 1)  # slow EMA smoothing factor
ema_slow = 0
num_periods_macd = 9  # MACD EMA time period
K_macd = 2 / (num_periods_macd + 1)  # MACD EMA smoothing factor
ema_macd = 0

ema_fast_values = []  # track fast EMA values for visualization purposes
ema_slow_values = []  # track slow EMA values for visualization purposes
macd_values = []  # track MACD values for visualization purposes
macd_signal_values = []  # MACD EMA values tracker
MACD_hist_values = []  # MACD - MACD-EMA
for close_price in df['Close']:
    if ema_fast == 0:  # first observation
        ema_fast = close_price
        ema_slow = close_price
    else:
        ema_fast = (close_price - ema_fast) * K_fast + ema_fast
        ema_slow = (close_price - ema_slow) * K_slow + ema_slow

    ema_fast_values.append(ema_fast)
    ema_slow_values.append(ema_slow)

    macd = ema_fast - ema_slow  # MACD is fast_MA - slow_EMA
    if ema_macd == 0:
        ema_macd = macd
    else:
        # signal is EMA of MACD values
        ema_macd = (macd - ema_macd) * K_slow + ema_macd

    macd_values.append(macd)
    macd_signal_values.append(ema_macd)
    MACD_hist_values.append(2 * (macd - ema_macd))

df = df.assign(ClosePrice=pd.Series(df['Close'], index=df.index))
df = df.assign(FastEMA10d=pd.Series(ema_fast_values, index=df.index))
df = df.assign(SlowEMA40d=pd.Series(ema_slow_values, index=df.index))
df = df.assign(MACD=pd.Series(macd_values, index=df.index))
df = df.assign(EMA_MACD20d=pd.Series(macd_signal_values, index=df.index))
df = df.assign(MACD_hist=pd.Series(MACD_hist_values, index=df.index))

close_price = df['ClosePrice']
ema_f = df['FastEMA10d']
ema_s = df['SlowEMA40d']
macd = df['MACD']
ema_macd = df['EMA_MACD20d']

# print(df[df['MACD_hist']>=0]['MACD_hist'])
# exit()
fig, ax = plt.subplots(2, 1)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

# 调整子图的间距，hspace表示高(height)方向的间距
plt.subplots_adjust(hspace=.1)

ax[0].set_ylabel('Close price in ￥')
ax[0].set_title('A_Stock %s MACD Indicator' % stock_code)
close_price.plot(ax=ax[0], color='g', lw=1., legend=True, use_index=False)
ema_f.plot(ax=ax[0], color='b', lw=1., legend=True, use_index=False)
ema_s.plot(ax=ax[0], color='r', lw=1., legend=True, use_index=False)
ax[0].yaxis.grid(True, which='major')

ax[1] = plt.subplot(212, sharex=ax[0])
macd.plot(ax=ax[1], color='k', lw=1., legend=True, sharex=ax[0], use_index=False)
ema_macd.plot(ax=ax[1], color='g', lw=1., legend=True, use_index=False)

# df[df['MACD_hist'] >= 0]['MACD_hist'].plot(ax=ax[2], color='r', kind='bar', legend=True, sharex=ax[0])
df['MACD_hist'].plot(ax=ax[1], color='r', kind='bar', legend=True, sharex=ax[0])
ax[1].yaxis.grid(True, which='major')

# 设置间隔，以便图形横坐标可以正常显示
interval = tick // 20
pl.xticks([i for i in range(1, tick + 1, interval)],
          [datetime.strftime(i, format='%Y-%m-%d %H:%M') for i in \
           pd.date_range(df.index[0], df.index[-1], freq='%dd' % (interval))],
          rotation=45)

plt.show()
