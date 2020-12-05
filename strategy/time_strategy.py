import pandas as pd
import numpy as np
import config

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 8000)
# pd.set_option('display.float_format', lambda x: '%.6f' % x)

df = pd.read_csv(config.input_data_path + '\\stock_data\\sh600000.csv', parse_dates=['date'])
df = df[['date', 'code', 'open', 'high', 'low', 'close', 'change']]
df.sort_values(by=['date'], inplace=True)
df['date'] = pd.to_datetime(df['date'])
df.reset_index(inplace=True, drop=True)

# 计算复权价
# df['change(除权)'] = df['close'].pct_change()
# print(df[abs(df['change'] - df['change(除权)']) > 0.00001])
# print(df);exit()
# 计算hfq的close
# 计算cp_factor
df['cp_factor'] = (df['change'] + 1).cumprod()
# 计算close的hfq价--->
# 等于用与该股票上市价格相等的钱，买入该股票后的资金曲线
initial_price = df.iloc[0]['close'] / (1 + df.iloc[0]['change'])  # 计算上市价格
df['close_hfq'] = initial_price * df['cp_factor']
# 计算hfq的open、high、low
df['open_hfq'] = df['open'] / df['close'] * df['close_hfq']
df['high_hfq'] = df['high'] / df['close'] * df['close_hfq']
df['low_hfq'] = df['low'] / df['close'] * df['close_hfq']

df[['open', 'high', 'low', 'close']] = df[['open_hfq', 'high_hfq', 'low_hfq', 'close_hfq']]
df = df[['date', 'code', 'open', 'high', 'low', 'close', 'change']]

# -----均线策略-----
# 当短期均线由下而上穿过长期均线时，第二天以open全仓买入并在之后一直持有股票
# 当短期均线由上而下穿过长期均线时，第二天以open卖出全部股票，并在之后一直空仓，直到下一次买入

# 计算移动均线(moving average(MA)) 并
ma_short = 5
ma_long = 50
df['ma_short'] = df['close'].rolling(ma_short, min_periods=1).mean()
df['ma_long'] = df['close'].rolling(ma_long, min_periods=1).mean()

# 补全缺失值(method_II)
# df['ma_short'].fillna(value=df['close'].expanding().mean(), inplace=True)
# df['ma_long'].fillna(value=df['close'].expanding().mean(), inplace=True)

# ---找出买入信号：
# 1.当天短期均线大于等于长期均线
condition_1 = df['ma_short'] >= df['ma_long']
# 2.上个交易日的短期均线小于长期均线
condition_2 = df['ma_short'].shift(1) < df['ma_long'].shift(1)
# 将买入信号当天的signal设置为1
df.loc[condition_1 & condition_2, 'signal'] = 1

# ---找出卖出信号：
# 1.当天短期均线小于等于长期均线
condition_1 = df['ma_short'] <= df['ma_long']
# 2.上个交易日的短期均线大于长期均线
condition_2 = df['ma_short'].shift(1) > df['ma_long'].shift(1)
# 将卖出信号当天的signal设置为1
df.loc[condition_1 & condition_2, 'signal'] = 0
# 删除无关变量
df.drop(['ma_short', 'ma_long'], axis=1, inplace=True)

# 由signal计算出实际每日持有股票仓位
# 计算仓位：收到信号后次日仓位情况才会发生变化。满仓=1，空仓=0
df['pos'] = df['signal'].shift()
df['pos'].fillna(method='ffill', inplace=True)
# 初始位置仓位情况补全为0
df['pos'].fillna(value=0, inplace=True)

# ---检查问题---
# 1.跌停时不得买卖股票
# 找开盘时涨停的日期：今日open相对于昨日涨了9.7%
cannot_buy = df['open'] > df['close'].shift(1) * 1.097
df.loc[cannot_buy & (df['pos'] == 1), 'pos'] = None
# position为空时，不能买卖，此时position只能和前一交易日保持一致
df['pos'].fillna(method='ffill', inplace=True)

# 截取上市一年后的交易日（约为250天），此时公司交易趋于稳定
df = df.iloc[250 - 1:]
# 第一天仓位设置为0，即从第二天开始买入股票
df.iloc[0, -1] = 0

# 计算实际资金曲线（simple）
# 在当天空仓时，pos=0，资产涨幅为0
# 在当天满仓时，pos=1，资产涨幅为股票本身的涨幅
df['equity_change'] = df['change'] * df['pos']
df['equity_curve'] = (1 + df['equity_change']).cumprod()

# 计算实际资金曲线（practical）
df = df[['date', 'code', 'open', 'high', 'low', 'close', 'change', 'pos']]
df.reset_index(inplace=True, drop=True)
# 设置参数
initial_money = 1000000  # 初始资金100万元
slippage = .01  # 滑点
c_rate = 5.0 / 10000  # 手续费，默认万分之5
t_rate = 1.0 / 1000  # 印花税，tax，默认千分之1

# 第一天的资金流动情况
df.at[0, 'hold_num'] = 0  # 持股数量，单位为股
df.at[0, 'stock_value'] = 0  # 持仓股价市值
df.at[0, 'actual_pos'] = 0  # 每日实际仓位情况
df.at[0, 'cash'] = initial_money  # 持有现金数
df.at[0, 'equity'] = initial_money  # 总资产 = 持仓股票市值 + 现金数

# 第一天之后每天的情况
# 从第二天起，逐行遍历，逐行计算
for i in range(1, df.shape[0]):
    # 前一天持有的股票数量
    hold_num = df.at[i - 1, 'hold_num']

    # 判断当天是否发生除权，若发生除权，需要调整hold_num
    # 若当天通过close计算出的change和当天的不同，说明当天发生了除权行为
    if abs((df.at[i, 'close'] / df.at[i - 1, 'close'] - 1) - df.at[i, 'change']) > 0.001:
        stock_value = df.at[i - 1, 'stock_value']
        # 交易所会公布除权之后的价格
        last_price = df.at[i, 'close'] / (1 + df.at[i, 'change'])
        hold_num = int(stock_value / last_price)

    # 判断是否需要调整仓位：比较今天、昨天的仓位position，不同则需要调整仓位
    # 需要调整仓位
    if df.at[i, 'pos'] != df.at[i - 1, 'pos']:
        # 需要调整仓位
        # 对于需要调整的仓位，需要买入多少股票
        # 需要持有的股票数 = 昨天的总资产 * 今天的仓位 / 今天的open
        theory_num = df.at[i - 1, 'equity'] * df.at[i, 'pos'] / df.at[i, 'open']
        # 对需要持有的股票数取整
        theory_num = int(theory_num)  # 向下取整

        # 比较theory_num与昨天持有股票数，判断加仓还是减仓
        # 加仓情形：
        if theory_num >= hold_num:
            # 计算实际需要买入的股票数量
            buy_num = theory_num - hold_num
            buy_num = int(buy_num / 100) * 100
            # 买入股票所需的现金(考虑滑点)
            buy_cash = buy_num * (slippage + df.at[i, 'open'])
            # 计算买入股票所花费的手续费，取两位小数
            commission = round(buy_cash * c_rate, 2)
            # 不足五元取五元
            if commission < 5 and commission != 0:
                commission = 5
            df.at[i, '手续费'] = commission
            # 计算当天收盘时持有的股票数量和现金
            df.at[i, 'hold_num'] = hold_num + buy_num
            df.at[i, 'cash'] = df.at[i - 1, 'cash'] - buy_cash - commission  # 手头剩余现金

        # 减仓情形：
        else:
            # 计算卖出股票的数量，卖出股票可以不是整数，不需要取整100。
            sell_num = hold_num - theory_num
            # 计算卖出股票得到的现金
            sell_cash = sell_num * (df.at[i, 'open'] - slippage)
            # 计算手续费，不足5元按5元收取，并保留两位小数
            commission = round(max(sell_cash * c_rate, 5), 2)
            df.at[i, '手续费'] = commission
            # 计算印花税，保留两位小数
            tax = round(sell_cash * t_rate, 2)
            df.at[i, '印花税'] = tax

            # 计算当天收盘时持有股票的数量和现金
            df.at[i, 'hold_num'] = hold_num - sell_num  # 持有股票数量
            df.at[i, 'cash'] = df.at[i - 1, 'cash'] + sell_cash - commission - tax

    # 不需要调仓
    else:
        # 计算当天收盘时持有股票的数量和现金
        df.at[i, 'hold_num'] = hold_num
        df.at[i, 'cash'] = df.at[i - 1, 'cash']  # 剩余现金

    # 以上计算得到每天的hold_num和cash
    # 计算当天收盘时的的各种资产数据
    df.at[i, 'stock_value'] = df.at[i, 'hold_num'] * df.at[i, 'close']  # 股票市值
    df.at[i, 'equity'] = df.at[i, 'cash'] + df.at[i, 'stock_value']  # 总资产
    df.at[i, 'actual_pos'] = df.at[i, 'stock_value'] / df.at[i, 'equity']  # 实际仓位

# df = df[['date', 'pos', 'cash', 'stock_value', 'equity', 'actual_pos']]
print(df)
