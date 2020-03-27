import pandas as pd
import config
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")


# 定义xgboost评估函数
# def accuracy(dtrain, y_pred):
#     y_true = dtrain.get_label()
#     return accuracy_score(y_true, y_pred)


# 加载特征
df_tfidf_lr = pd.read_csv(config.feat_path + '/tfidf/lr_prob_education.csv')
df_tfidf_mnb = pd.read_csv(config.feat_path + '/tfidf/mnb_prob_education.csv')
df_tfidf_svm = pd.read_csv(config.feat_path + '/tfidf/svm_prob_education.csv')
df_w2v = pd.read_csv(config.feat_path + '/w2v/w2v_avg.csv')

# 加载标签
df_lb = pd.read_csv(config.corpus_file, usecols=['ID', 'Education'], encoding='utf-8',
                    nrows=config.train_num)

tr_num = config.cv_train_num
df = pd.concat([df_tfidf_lr, df_tfidf_mnb, df_tfidf_svm, df_w2v], axis=1)
# print(df.columns)
x = df.iloc[:tr_num]
y = df_lb['Education'][:tr_num]
x_te = df.iloc[tr_num:]
y_te = df_lb['Education'][tr_num:]

df_sub = pd.DataFrame()
df_sub['ID'] = df_lb.iloc[tr_num:]['ID']

# 参数设置
num_education_class = len(pd.value_counts(df_lb['Education']))
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类
    'stratified': True,
    'num_class': num_education_class,  # 类别数
    'max_depth': config.max_depth,  # 构建树的深度，越大越容易过拟合
    'min_child_weight': config.min_child_weight,  # 若叶子节点的样本权重和小于min_child_weight则拆分过程结束
    'subsample': config.subsample,  # 每棵树对样本的采样率(行采样)，默认为1
    'colsample_bytree': config.colsample_bytree,  # 每棵树列采样率，也就是特征采样率，默认是1，一般设为0.8
    'gamma': config.gamma,  # 指定了节点分裂所需的最小损失函数下降值，越大越保守，一般0.1 0.2的样子，默认为0
    'lambda': config.lam,  # 权重值的L2正则化项，参数越大，模型越不容易过拟合，默认为0
    'eta': config.eta,  # 学习率，通常设置为0.01-0.2，默认为0.3
    'silent': config.silent,  # 设置成1，则没有运行信息输入
    'scale_pos_weight': config.scale_pos_weight,  # 类别不平衡的情况下，将参数设置大于0，可以加快收敛，默认是1
    'seed': config.seed,
}
dtrain = xgb.DMatrix(x, y)
dtest = xgb.DMatrix(x_te, y_te)
watchlist = [(dtrain, 'train'), (dtest, 'test')]
bst = xgb.train(params, dtrain, config.num_boost_round, evals=watchlist, maximize=True,
                early_stopping_rounds=config.early_stopping_rounds, verbose_eval=config.verbose_eval)

y_pred = bst.predict(dtest)
accuracy = accuracy_score(y_te, y_pred)
print('测试集学历的准确率: %.3f' % accuracy)

df_sub['Education'] = (bst.predict(dtest)).astype(int)
df_sub['ID'] = df_sub['ID'].astype(str)
df_sub.to_csv(config.output + '/education_result.csv', encoding='utf-8', index=False)

'''
    num_boost_round:迭代的次数，默认是10
    evals：这是一个列表，用于对训练过程中进行评估列表中的元素。
    形式是evals = [(dtrain,'train'),(dval,'val')] 或者是 evals =[(dtrain,'train')]
    feval：自定义评估函数
    maximize：是否对评估函数进行最大化
    early_stopping_rounds：早停次数
    verbose_eval：可以输入布尔型或数值型，若为True,则对evals中元素的评估结果会输出在结果中；
    如果输入数字，假设为5，则每隔5个迭代输出一次

'''
