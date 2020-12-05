import sys

sys.path.append('..')
import config, os, time
import pandas as pd
import numpy as np

pd.set_option('expand_frame_repr', False)
pd.set_option('io.hdf.default_format', 'table')
start = time.perf_counter()

input_path = os.path.join(config.input_data_path, 'stock_data')
# input_path = os.path.abspath(os.oath.join(config.input_data_path, 'stock_minute_data'))

# print(config.input_data_path);exit()

f_list = []

for root, dirs, files in os.walk(input_path):
    for f_name in files:
        if f_name.endswith('.csv'):
            f_list.append(os.path.abspath(os.path.join(root, f_name)))

hdf_db = pd.HDFStore(os.path.join(config.input_data_path, 'stock_db.h5'), mode='w', complevel=6, complib='blosc')


def import_to_hdf(f_path):
    stock_code = f_path.split('/')[-1].split('.')[0]
    hdf_db.put(key=stock_code, value=pd.read_csv(f_path))
    print(stock_code, '已导入')


if __name__ == '__main__':

    for f_path in sorted(f_list):
        import_to_hdf(f_path)
        # break

hdf_db.close()
print('总用时{:.5f}s'.format(time.perf_counter() - start))
