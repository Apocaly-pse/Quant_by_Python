import sys

sys.path.append('..')
import config, os
import matplotlib as mpl

mpl.use('TkAgg')

import pandas as pd
import mplfinance as mpf
from cycler import cycler
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

plt.style.use("ggplot")

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

df = import_csv(symbol, period)[-150:]


# 参数设置
kwargs = dict(type='candle', mav=(7, 30, 60), volume=True, title='\nA_stock %s, %s candle_line' % (symbol, period),
              ylabel='OHLC Candles', ylabel_lower='Shares\nTraded Volume', figratio=(15, 10), figscale=1)

mc = mpf.make_marketcolors(up='r', down='g', wick='i', edge={'up':'red', 'down':'green'}, volume='i', ohlc='i', )
s = mpf.make_mpf_style(gridaxis='both', gridstyle='-.', y_on_right=False, marketcolors=mc)

mpl.rcParams['axes.prop_cycle'] = cycler(
    color=['dodgerblue', 'deeppink', 'navy', 'teal', 'maroon', 'darkorange', 'indigo'])
mpl.rcParams['lines.linewidth'] = .5

labels = ['MA7', 'MA30', 'MA60']
color=['dodgerblue', 'deeppink', 'navy',]
patches = [ mpatches.Patch(color=color[i], label="{:s}".format(labels[i]) ) for i in range(len(color)) ]

fig = plt.figure()
ax=fig.add_subplot(1,1,1)
ax.legend(handles=patches, bbox_to_anchor=(.99,1.1), loc='best')
# mpf.plot(df, style=s, savefig='A_stock-%s %s_candle_line' % (symbol, period) + '.png', **kwargs)


# apd=mpf.make_addplot()
mpf.plot(df, style=s, **kwargs)



plt.show()

