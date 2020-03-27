import pandas as pd
from config import *
import utils
import warnings
import os

pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

if not os.path.exists(processed_data_file):
    print('开始预处理数据......')
    raw_train_data = pd.read_csv(train_data, sep="###__###", header=None, encoding='utf-8', nrows=train_num)
    raw_train_data.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']

    # 分词处理
    raw_train_data['Query_List'] = raw_train_data['Query_List'].apply(
        lambda x: utils.split_word(x, stopwords_file))
    # print(raw_train_data.head())

    raw_test_data = pd.read_csv(test_data, sep="###__###", header=None, encoding='utf-8', nrows=test_num)
    raw_test_data.columns = ['ID', 'Query_List']
    # print(raw_test_data.shape)

    # 分词处理
    raw_test_data['Query_List'] = raw_test_data['Query_List'].apply(
        lambda x: utils.split_word(x, stopwords_file))
    # print(raw_test_data.head())

    # 写出数据
    df_all = pd.concat([raw_train_data, raw_test_data]).fillna(0)  # 默认上下合并
    print(df_all.shape)
    df_all.to_csv(processed_data + 'all_data.csv', index=False)
