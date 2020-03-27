import pandas as pd
from config import *
import utils
from gensim.models import Word2Vec
import warnings

warnings.filterwarnings("ignore")

# 读取数据
df_all = pd.read_csv(processed_data_file, encoding='utf-8', nrows=train_num)

contents = list(df_all['Query_List'].values)
words = [[word for word in content.split(' ')] for content in contents]
# 过滤低频词
words_list = utils.filter_words(words)
print(words_list[:2]) # [[],[]]

print('开始训练Word2Vec模型......')
w2v = Word2Vec(words_list, size=w2v_dim, window=5, iter=15, workers=10, seed=seed)
w2v.save(model_path + 'w2v.model')
print('已完成！')

'''
    size:词向量的维度，默认为100
    window：窗口大小，即词向量上下文最大距离，默认为5。window越大，则和某一词较远的词也会产生上下文关系
    min_count: 需要计算词向量的最小词频，默认是5
    iter: 随机梯度下降法中迭代的最大次数，默认是5
    workers：用于控制训练的并行数
'''
