import pandas as pd
from config import *
import utils
import numpy as np
from gensim.models import Word2Vec
import warnings

warnings.filterwarnings("ignore")

# 加载Word2Vec模型
model = Word2Vec.load(model_file)

# 读取数据
df_all = pd.read_csv(processed_data_file, encoding='utf-8', nrows=train_num)

contents = list(df_all['Query_List'].values)
words = [[word for word in content.split(' ')] for content in contents]
words_list = utils.filter_words(words)

print('开始构造word2vec特征......')
w2v_feat = np.zeros((len(words_list), w2v_dim))  # w2v_dim = 300  # 词向量的维度
w2v_feat_avg = np.zeros((len(words_list), w2v_dim))

i = 0
for words in words_list:
    num = 0
    for word in words:
        vec = model[word]
        w2v_feat[i, :] += vec  # 属于同一文档的词向量直接相加
        num += 1

    w2v_feat_avg[i, :] = w2v_feat[i, :] / num  # 属于同一个文档的词向量加权平均
    i += 1

pd.DataFrame(w2v_feat).to_csv(feat_path + 'w2v/w2v.csv', encoding='utf-8', index=False)
pd.DataFrame(w2v_feat_avg).to_csv(feat_path + 'w2v/w2v_avg.csv', encoding='utf-8', index=False)
print('已完成！')
