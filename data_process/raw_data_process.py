import sys
sys.path.append('..')
import config
import os, time
import pandas as pd
import numpy as np
# from multiprocessing import Pool


pd.set_option('expand_frame_repr', False)
pd.set_option('io.hdf.default_format', 'table')
start = time.perf_counter()

# input_path = config.input_data_path + '\\r_stock_data'
# output_path = config.input_data_path + '\\stock_data\\'
# # print(path);exit()
# f_list = []
#
# for root, dirs, files in os.walk(input_path):
#     for fname in files:
#         if fname.endswith('.csv'):
#             fpath = os.path.join(root, fname)
#             f_list.append(fpath)
# # print(f_list);exit()
# # all_data = pd.DataFrame()
# for f_path in sorted(f_list):
#     df = pd.read_csv(f_path)
#     df['pre_close'] = None
#     df['buy1'] = None
#     df['sell1'] = None
#     df = df[['code', 'date', 'open', 'high', 'low', 'close', 'change', 'volume', 'money', 'traded_market_value',
#              'market_value', 'turnover', 'pre_close', 'adjust_price', 'adjust_price_f', 'buy1', 'sell1']]
#     df.sort_values(by=['date'], inplace=True)
#     # all_data = all_data.append(df, ignore_index=True)
#     # print(all_data);exit()
#     df.to_csv(output_path + f_path[-12:], index=False)
#     print(f_path[-12:], '_ok')
#     break
#     # time.sleep(.5)
input_path = config.input_data_path + '\\stock_data'
output_path = config.input_data_path

f_list = []

for root, dirs, files in os.walk(input_path):
    for f_name in files:
        if f_name.endswith('.csv'):
            f_list.append(os.path.abspath(os.path.join(root, f_name)))
# print(f_list[0]);exit()
hdf_db = pd.HDFStore(output_path + '\\stock_db.h5', mode='w', complevel=5, complib='blosc')

def import_to_hdf(f_path):
    stock_code = f_path.split('\\')[-1].split('.')[0]
    # pd.read_csv(f_path).to_hdf(output_path + '\\stock_db.h5', key=stock_code)
    hdf_db.put(key=stock_code, value=pd.read_csv(f_path))
    print(stock_code, '已导入')

if __name__ == '__main__':
    # pool=Pool()
    # pool.map(import_to_hdf, [f_path for f_path in sorted(f_list)])
    for f_path in sorted(f_list):
        import_to_hdf(f_path)

# hdf_db.close()
print('总用时{:.5f}s'.format(time.perf_counter() - start))
