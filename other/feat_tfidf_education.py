import pandas as pd
import config
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

# 读取数据
df_all = pd.read_csv(config.corpus_file, encoding='utf-8', nrows=config.train_num)

# tf-idf
vec = TfidfVectorizer(min_df=3, max_df=0.95)
tf_idf = vec.fit_transform(df_all['Query_List'])
# print(tf_idf.shape)  # (11000, 72735)


print('LR开始建模......')
tr_num = config.cv_train_num
num_education_class = len(pd.value_counts(df_all['Education']))
# print(num_education_class)  # 3
n = 5

# 划分数据集
x = tf_idf[:tr_num]  # (0, 27600)	0.043404722639893076
y = df_all['Education'][:tr_num]
x_te = tf_idf[tr_num:]
y_te = df_all['Education'][tr_num:]


stack = np.zeros((x.shape[0], num_education_class))
# print(stack.shape)
stack_te = np.zeros((x_te.shape[0], num_education_class))
# print(stack_te.shape)

score_va = 0  # 验证集得分
score_te = 0

for i, (tr, va) in enumerate(StratifiedKFold(n_splits=n, random_state=0).split(x, y)):
    clf = LogisticRegression(C=2)
    clf.fit(x[tr], y[tr])
    y_pred_va = clf.predict_proba(x[va])
    y_pred_te = clf.predict_proba(x_te)

    # 计算得分
    score_va += accuracy_score(y[va], clf.predict(x[va]))
    score_te += accuracy_score(y_te, clf.predict(x_te))

    # 得分累加
    stack[va] += y_pred_va
    stack_te += y_pred_te

# 计算平均得分
score_va /= n
score_te /= n
print('验证集学历的平均准确率:%f' % score_va)
print('测试集学历的平均准确率:%f' % score_te)

stack_te /= n  # 归一化
stack_all = np.vstack([stack, stack_te])
df_stack = pd.DataFrame(index=range(len(df_all)))
for i in range(stack_all.shape[1]):
    df_stack['tfidf_lr_{}'.format(i)] = stack_all[:, i]

df_stack.to_csv(config.feat_path + '/tfidf/lr_prob_education.csv', index=False, encoding='utf-8')
# 交叉验证
# scores = cross_val_score(LogisticRegression(C=2), x, y, cv=StratifiedKFold(n_splits=3))
# print(scores.mean())
print('**********************************************************************************************')
print('Naive Bayes开始建模......')
tr_num = config.cv_train_num
n = 5

x = tf_idf[:tr_num]
y = df_all['Education'][:tr_num]
x_te = tf_idf[tr_num:]
y_te = df_all['Education'][tr_num:]

stack = np.zeros((x.shape[0], num_education_class))
stack_te = np.zeros((x_te.shape[0], num_education_class))

score_va = 0  # 验证集得分
score_te = 0  # 测试集得分

# 交叉验证
for i, (tr, va) in enumerate(StratifiedKFold(n_splits=n, random_state=config.seed).split(x, y)):
    clf = MultinomialNB()
    clf.fit(x[tr], y[tr])
    y_pred_va = clf.predict_proba(x[va])
    y_pred_te = clf.predict_proba(x_te)

    # 计算得分
    score_va += accuracy_score(y[va], clf.predict(x[va]))
    score_te += accuracy_score(y_te, clf.predict(x_te))

    # 得分累加
    stack[va] += y_pred_va
    stack_te += y_pred_te

# 计算平均得分
score_va /= n
score_te /= n
print('验证集学历的平均准确率:%f' % score_va)
print('测试集学历的平均准确率:%f' % score_te)

stack_te /= n  # 归一化
stack_all = np.vstack([stack, stack_te])
df_stack = pd.DataFrame(index=range(len(df_all)))
for i in range(stack_all.shape[1]):
    df_stack['tfidf_mnb_{}'.format(i)] = stack_all[:, i]

df_stack.to_csv(config.feat_path + '/tfidf/mnb_prob_education.csv', index=False, encoding='utf-8')

print('**********************************************************************************************')
print('SVM开始建模......')
tr_num = config.cv_train_num
n = 5

x = tf_idf[:tr_num]
y = df_all['Education'][:tr_num]
x_te = tf_idf[tr_num:]
y_te = df_all['Education'][tr_num:]

stack = np.zeros((x.shape[0], num_education_class))
stack_te = np.zeros((x_te.shape[0], num_education_class))

score_va = 0  # 验证集得分
score_te = 0  # 测试集得分

# 交叉验证
for i, (tr, va) in enumerate(StratifiedKFold(n_splits=n, random_state=config.seed).split(x, y)):
    clf = svm.LinearSVC(loss='hinge', tol=0.000001, C=0.5, random_state=config.seed, max_iter=1000)
    clf.fit(x[tr], y[tr])
    y_pred_va = clf.decision_function(x[va])
    y_pred_te = clf.decision_function(x_te)

    # 计算得分
    score_va += accuracy_score(y[va], clf.predict(x[va]))
    score_te += accuracy_score(y_te, clf.predict(x_te))

    # 得分累加
    stack[va] += y_pred_va
    stack_te += y_pred_te

# 计算平均得分
score_va /= n
score_te /= n
print('验证集学历的平均准确率:%f' % score_va)
print('测试集学历的平均准确率:%f' % score_te)

stack_te /= n  # 归一化
stack_all = np.vstack([stack, stack_te])
df_stack = pd.DataFrame(index=range(len(df_all)))
for i in range(stack_all.shape[1]):
    df_stack['tfidf_svm_{}'.format(i)] = stack_all[:, i]

df_stack.to_csv(config.feat_path + '/tfidf/svm_prob_education.csv', index=False, encoding='utf-8')
