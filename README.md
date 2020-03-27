# user-portrait

数据预处理：分词、去停用词

特征构建：TFIDF、Word2Vec、统计特征

模型融合：stacking思想，两层模型结构。

第一层：传痛的机器学习模型LR、MultinomialNB、SVM来训练tfidf特征；基于Word2Vec的特征，将词语表示成一个固定长度的向量。对于Word2Vec模型，选取的维数为300，将频数低于5的词语过滤掉。
    
第二层：XGBoost模型训练Word2Vec、统计特征和第一层模型传来的概率特征。
