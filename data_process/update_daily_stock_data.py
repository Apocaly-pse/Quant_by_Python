#!/usr/bin/python
import sys, os

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
import config, re, time, json, requests
from datetime import datetime
import pandas as pd

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
start = time.perf_counter()


def get_content_from_internet(url, max_try_num=10, sleep_time=5):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36"
    }
    # 抓取函数
    get_success = False  # 是否成功抓取到内容
    # 抓取内容
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


# 从新浪获取指定股票的数据
def get_today_data_from_sinajs(code_list):
    # 构建url
    url = "http://hq.sinajs.cn/list=" + ",".join(code_list)

    # 抓取数据
    content = get_content_from_internet(url)
    # content = content.decode('gbk')

    # 将数据转换成DataFrame
    content = content.strip()  # 去掉文本前后的空格、回车等
    data_line = content.split('\n')  # 每行是一个股票的数据
    data_line = [i.replace('var hq_str_', '').split(',') for i in data_line]
    df = pd.DataFrame(data_line, dtype='float')

    # 对DataFrame进行整理
    df[0] = df[0].str.split('="')
    df['code'] = df[0].str[0].str.strip()
    # df['stock_name'] = df[0].str[-1].str.strip()
    df['candle_end_time'] = df[30] + ' ' + df[31]  # 股票市场的K线，是普遍以当跟K线结束时间来命名的
    df['candle_end_time'] = pd.to_datetime(df['candle_end_time'])
    rename_dict = {1: 'open', 2: 'pre_close', 3: 'close', 4: 'high', 5: 'low', 6: 'buy1', 7: 'sell1',
                   8: 'volume', 9: 'money', 32: 'status'}
    # 其中volume单位是手，money单位是元
    df.rename(columns=rename_dict, inplace=True)
    df['status'] = df['status'].astype(str).str.strip('";')
    df = df[['code', 'candle_end_time', 'open', 'high', 'low', 'close', 'pre_close', 'volume',
             'money', 'buy1', 'sell1', 'status']]

    return df


# 判断今天是否是交易日
def is_today_trading_day():
    # 如果是返回True，否则返回False
    # 获取上证指数今天的数据
    df = get_today_data_from_sinajs(code_list=['sh000001'])
    sh_date = df.iloc[0]['candle_end_time']  # 上证指数最近交易日

    # 判断今天日期和sh_date是否相同
    return datetime.now().date() == sh_date.date()


# =====函数：从新浪获取所有股票的数据
def get_all_today_stock_data_from_sina_marketcenter():
    """
    http://vip.stock.finance.sina.com.cn/mkt/#stock_hs_up
    从新浪网址的上述的网址，逐页获取最近一个交易日所有股票的数据
    并返回一个存储股票数据的DataFrame
    """

    # 数据网址
    raw_url = 'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=%s&num=80&sort=symbol&asc=1&node=hs_a&symbol=&_s_r_a=sort'
    page_num = 1

    # ===存储数据的DataFrame
    all_df = pd.DataFrame()

    # ===获取上证指数最近一个交易日的日期。
    df = get_today_data_from_sinajs(code_list=['sh000001'])
    sh_date = df.iloc[0]['candle_end_time'].date()  # 上证指数最近交易日

    # ===开始逐页遍历，获取股票数据
    while True:
        # 构建url
        url = raw_url % (page_num)
        print('开始抓取页数：', page_num)

        # 抓取数据
        content = get_content_from_internet(url)

        # 判断页数是否为空
        if eval(content) == []:
            print('抓取到页数的尽头，退出循环')
            break

        # 通过正则表达式，给key加上引号
        content = re.sub(r'(?<={|,)([a-zA-Z]\w*)(?=:)', r'"\1"', content)
        # 将数据转换成dict格式
        content = json.loads(content)
        # 将数据转换成DataFrame格式
        df = pd.DataFrame(content, dtype='float')

        # 对数据进行整理
        # 重命名
        rename_dict = {'code': 'code_num', 'symbol': 'code', 'trade': 'close', 'settlement': 'pre_close',
                       'changepercent': 'change', 'amount': 'money', 'buy': 'buy1', 'sell': 'sell1',
                       'nmc': 'traded_market_value', 'mktcap': 'market_value', 'turnoverratio': 'turnover'}
        df.rename(columns=rename_dict, inplace=True)
        # 添加交易日期
        df['date'] = pd.to_datetime(sh_date)

        df['adjust_price'] = None
        df['adjust_price_f'] = None

        df['turnover'] = df['turnover'].apply(lambda x: x * 0.01)
        df['change'] = df['change'].apply(lambda x: x * 0.01)
        df['traded_market_value'] = df['traded_market_value'].apply(lambda x: x * (10 ** 4))
        df['market_value'] = df['market_value'].apply(lambda x: x * (10 ** 4))

        # # 后复权
        # # 等于用与该股票上市价格相等的钱，买入该股票后的资金曲线
        # hfq = (df['change'] + 1).cumprod()
        # initial_price = df.iloc[0]['close'] / (1 + df.iloc[0]['change'])  # 计算上市价格
        # df['adjust_price'][df.shape[0] - 1] = initial_price * hfq[0]
        #
        # # 前复权
        # # 等于用与该股票最新收盘价相等的钱，买入该股票后向前计算得到的资金曲线
        # df.sort_values(by=['date'], ascending=0, inplace=True)
        # qfq = (1 / (df['change'] + 1)).cumprod()
        # after_price = df.iloc[0]['close'] * (1 + df['change'])
        # df['adjust_price_f'][df.shape[0] - 1] = after_price[0] * qfq[0]
        # df.sort_values(by=['date'], inplace=True)

        # 取需要的列
        df = df[['code', 'date', 'open', 'high', 'low', 'close', 'change', 'volume', 'money',
                 'traded_market_value', 'market_value', 'turnover', 'pre_close', 'adjust_price', 'adjust_price_f',
                 'buy1', 'sell1']]

        # 合并数据
        all_df = all_df.append(df, ignore_index=True)

        # 将页数+1
        page_num += 1
        time.sleep(1)

        # break

    # ===删除当天停盘的股票
    all_df = all_df[all_df['open'] - 0 >= 0.00001]
    all_df.reset_index(drop=True, inplace=True)

    return all_df


if __name__ == '__main__':
    # 判断今天是否是交易日
    if is_today_trading_day() is False:
        print('今天不是交易日，不需要更新股票数据，退出程序')
        exit()

    # 判断当前时间是否超过15点
    if datetime.now().hour < 15:  # 保险起见可以小于16点
        print('今天股票尚未收盘，不更新股票数据，退出程序')
        exit()

    # 获取今天所有的股票数据
    df = get_all_today_stock_data_from_sina_marketcenter()

    # 对数据进行存储
    for i in df.index:
        t = df.iloc[i:i + 1, :]
        stock_code = t.iloc[0]['code']

        # 构建存储文件路径
        path_csv = os.path.join(config.input_data_path, 'stock_data', stock_code + '.csv')
        # 文件存在，不是新股
        if os.path.exists(path_csv):
            t.to_csv(path_csv, index=False, header=None, mode='a')
        # 文件不存在，说明是新股
        else:
            t.to_csv(path_csv, index=False, mode='w')
        print(stock_code)

    print('运行总用时%.3fs' % (time.perf_counter() - start))
