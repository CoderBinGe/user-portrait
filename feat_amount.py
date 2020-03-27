import pandas as pd
from config import *
import warnings

warnings.filterwarnings("ignore")

raw_train_data = pd.read_csv(train_data, sep="###__###", header=None, encoding='utf-8',
                             nrows=train_num)
raw_train_data.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']
# print(raw_train_data.head())

raw_train_data['query_len']=raw_train_data.Query_List.agg(lambda s : len(s))
print(raw_train_data.head())





