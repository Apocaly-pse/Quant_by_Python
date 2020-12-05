import sys
import matplotlib as mpl
import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

mpl.use('TkAgg')
sys.path.append('..')


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
    df.set_index(['Date'], inplace=True)
    return df

symbol = 'sh600100'
period = 'day'

stock_df = import_csv(symbol, period)[1800:5000]
stock_df['Open-Close'] = stock_df.Open - stock_df.Close
stock_df['High-Low'] = stock_df.High - stock_df.Low
stock_df = stock_df.dropna()
X = stock_df[['Open-Close', 'High-Low']]
Y = np.where(stock_df['Close'].shift(-1) > stock_df['Close'], 1, -1)
# print(X);exit()
split_ratio = 0.8
split_value = int(split_ratio * len(stock_df))
X_train = X[:split_value]
Y_train = Y[:split_value]
X_test = X[split_value:]
Y_test = Y[split_value:]

logistic = LogisticRegression()
logistic.fit(X_train, Y_train)
accuracy_train = accuracy_score(Y_train, logistic.predict(X_train))
accuracy_test = accuracy_score(Y_test, logistic.predict(X_test))
print(accuracy_train, accuracy_test)

stock_df['Predicted_Signal'] = logistic.predict(X)
stock_df['%s_Returns'%(symbol)] = np.log(stock_df['Close'] / stock_df['Close'].shift(1))


def calculate_return(df, split_value, symbol):
    cum_real_return = df[split_value:]['%s_Returns' % symbol].cumsum() * 100
    df['Strategy_Returns'] = df['%s_Returns' % symbol] * df['Predicted_Signal'].shift(1)
    return cum_real_return


def calculate_strategy_return(df, split_value):
    cum_strategy_return = df[split_value:]['Strategy_Returns'].cumsum() * 100
    return cum_strategy_return


cum_real_return = calculate_return(stock_df, split_value=len(X_train), symbol=symbol)
cum_strategy_return = calculate_strategy_return(stock_df, split_value=len(X_train))


def plot_shart(cum_symbol_return, cum_strategy_return, symbol):
    plt.figure(figsize=(10, 5))
    plt.plot(cum_symbol_return, label='%s Returns' % symbol)
    plt.plot(cum_strategy_return, label='Strategy Returns')
    plt.legend()
    plt.show()


plot_shart(cum_real_return, cum_strategy_return, symbol=symbol)


def sharpe_ratio(symbol_returns, strategy_returns):
    strategy_std = strategy_returns.std()
    sharpe = (strategy_returns - symbol_returns) / strategy_std
    return sharpe.mean()

print(sharpe_ratio(cum_strategy_return,cum_real_return))
