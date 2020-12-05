#!/usr/bin/python

from random import randint
import json, requests, os, sys, time
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
import config
import pandas as pd
from multiprocessing import Pool

start = time.perf_counter()
pd.set_option('expand_frame_repr', False)


# 生成随机数用于构造url
def r_num(n=16):
    return str(randint(10 ** (n - 1), (10 ** n) - 1))


input_path = os.path.join(config.input_data_path, 'stock_data')
f_list = []
for root, dirs, files in os.walk(input_path):
    for fname in files:
        if fname.endswith('.csv'):
            f_list.append(fname.split('.')[0])
# print(f_list);exit()

def get_content(url, max_try_num=10, sleep_time=5):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36"}
    get_success = False  # 是否成功抓取到内容
    for i in range(max_try_num):
        try:
            content = requests.get(url=url, timeout=10).text
            get_success = True  # 成功抓取到内容
            break
        except Exception as e:
            print('抓取报错，次数：', i + 1, '内容：', e)
            time.sleep(sleep_time)
    # 判断是否成功抓取内容
    if get_success:
        return content
    else:
        raise ValueError('抓取不断报错，达到尝试上限!!')


def get_today_data_from_sinajs(code_list):
    # 将数据转换成DataFrame
    data_line = get_content("http://hq.sinajs.cn/list=" + ",".join(code_list)).strip().split('\n')  # 每行是一个股票的数据
    data_line = [i.replace('var hq_str_', '').split(',') for i in data_line]
    df = pd.DataFrame(data_line, dtype='float')

    # 对DataFrame进行整理
    df[0] = df[0].str.split('="')
    df['code'] = df[0].str[0].str.strip()
    df['candle_end_time'] = df[30] + ' ' + df[31]  # 股票市场的K线，是普遍以当跟K线结束时间来命名的
    df['candle_end_time'] = pd.to_datetime(df['candle_end_time'])

    return df


# 判断今天是否是交易日
def is_today_trading_day():
    # 如果是返回True，否则返回False
    df = get_today_data_from_sinajs(code_list=['sh000001'])
    sh_date = df.iloc[0]['candle_end_time']  # 上证指数最近交易日
    # 判断今天日期和sh_date是否相同
    return datetime.now().date() == sh_date.date()


# 判断今天是否是交易日
if is_today_trading_day() is False:
    print('今天不是交易日，不需要更新股票数据，退出程序')
    exit()

# 判断当前时间是否超过15点
if datetime.now().hour < 15:  # 保险起见可以小于16点
    print('今天股票尚未收盘，不更新股票数据，退出程序')
    exit()


def candle_minute_line(stock_code, k_type=1, num=241):
    url = 'http://ifzq.gtimg.cn/appstock/app/kline/mkline?param=%s,m%s,,%d&var=m%s_today&r=0.%s' % (
    stock_code, k_type, num, k_type, r_num())
    data = get_content(url)
    # 处理json数据->dict-DataFrame
    k_data = json.loads(data.split('=', maxsplit=1)[-1])['data'][stock_code]['m' + str(k_type)]
    if k_data:
        # volume单位为手(1手=100股)
        df = pd.DataFrame(k_data)
        df.rename(columns={0: 'candle_end_time', 1: 'open', 2: 'close', 3: 'high', 4: 'low', 5: 'volume'}, inplace=True)
        # 处理时间
        df['candle_end_time'] = df['candle_end_time'].apply(
            lambda x: '%s-%s-%s %s:%s' % (x[0:4], x[4:6], x[6:8], x[8:10], x[10:12]))
        df['candle_end_time'] = pd.to_datetime(df['candle_end_time'])
        df = df[['candle_end_time', 'open', 'close', 'high', 'low', 'volume']]
    else:
        df = None
    return df


def save_to_file(code):
    path_csv = os.path.join(config.input_data_path, 'stock_minute_data', code + '.csv')
    if candle_minute_line(code) is not None:
        if os.path.exists(path_csv):
            candle_minute_line(code).to_csv(path_csv, index=False, header=None, mode='a')
        else:
            candle_minute_line(code).to_csv(path_csv, index=False, mode='w')
        print(code, 'ok!')


if __name__ == '__main__':
    pool = Pool()
    pool.map(save_to_file, [code for code in f_list])
# for code in ['sh688566']:
# 	save_to_file(code)

print('update_minute_data_Costs:%.3fs' % (time.perf_counter() - start))
