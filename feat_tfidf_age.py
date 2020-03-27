import pandas as pd
from config import *
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
df_all = pd.read_csv(processed_data_file, encoding='utf-8', nrows=train_num)

# tf-idf
vec = TfidfVectorizer(min_df=3, max_df=0.95, sublinear_tf=True)
tf_idf = vec.fit_transform(df_all['Query_List'])

print('LR开始建模......')

# 训练集
x = tf_idf[:cv_train_num]
y = df_all['Age'][:cv_train_num]
# 测试集
x_test = tf_idf[cv_train_num:]
y_test = df_all['Age'][cv_train_num:]

num_age_class = len(pd.value_counts(df_all['Age']))
# print(num_age_class)  # 7

stack = np.zeros((x.shape[0], num_age_class))
# print(stack.shape)
stack_te = np.zeros((x_test.shape[0], num_age_class))
# print(stack_te.shape)

score_va = 0  # 验证集得分
score_te = 0  # 测试集得分

# 交叉验证
for i, (train, dev) in enumerate(StratifiedKFold(n_splits=n, random_state=0).split(x, y)):
    clf = LogisticRegression(C=2)
    clf.fit(x[train], y[train])
    y_pred_dev = clf.predict_proba(x[dev])
    y_pred_test = clf.predict_proba(x_test)

    # 计算得分
    score_va += accuracy_score(y[dev], clf.predict(x[dev]))
    score_te += accuracy_score(y_test, clf.predict(x_test))

    # 得分累加
    stack[dev] += y_pred_dev
    stack_te += y_pred_test

# 计算平均得分
score_va /= n
score_te /= n
print('验证集年龄的准确率:%f' % score_va)
print('测试集年龄的准确率:%f' % score_te)

stack_te /= n  # 归一化
stack_all = np.vstack([stack, stack_te])
df_stack = pd.DataFrame(index=range(len(df_all)))
for i in range(stack_all.shape[1]):
    df_stack['tfidf_lr_{}'.format(i)] = stack_all[:, i]

df_stack.to_csv(feat_path + 'tfidf/lr_prob_age.csv', index=False, encoding='utf-8')
# 交叉验证
# scores = cross_val_score(LogisticRegression(C=2), x, y, cv=StratifiedKFold(n_splits=3))
# print(scores.mean())
print('**********************************************************************************************')
print('Naive Bayes开始建模......')

# 训练集
x = tf_idf[:cv_train_num]
y = df_all['Age'][:cv_train_num]
# 测试集
x_test = tf_idf[cv_train_num:]
y_test = df_all['Age'][cv_train_num:]

num_age_class = len(pd.value_counts(df_all['Age']))
# print(num_Age_class)  # 3

stack = np.zeros((x.shape[0], num_age_class))
# print(stack.shape)
stack_te = np.zeros((x_test.shape[0], num_age_class))
# print(stack_te.shape)

score_va = 0  # 验证集得分
score_te = 0  # 测试集得分

# 交叉验证
for i, (train, dev) in enumerate(StratifiedKFold(n_splits=n, random_state=0).split(x, y)):
    clf = MultinomialNB()
    clf.fit(x[train], y[train])
    y_pred_dev = clf.predict_proba(x[dev])
    y_pred_test = clf.predict_proba(x_test)

    # 计算得分
    score_va += accuracy_score(y[dev], clf.predict(x[dev]))
    score_te += accuracy_score(y_test, clf.predict(x_test))

    # 得分累加
    stack[dev] += y_pred_dev
    stack_te += y_pred_test

# 计算平均得分
score_va /= n
score_te /= n
print('验证集年龄的准确率:%f' % score_va)
print('测试集年龄的准确率:%f' % score_te)

stack_te /= n  # 归一化
stack_all = np.vstack([stack, stack_te])
df_stack = pd.DataFrame(index=range(len(df_all)))
for i in range(stack_all.shape[1]):
    df_stack['tfidf_mnb_{}'.format(i)] = stack_all[:, i]

df_stack.to_csv(feat_path + 'tfidf/mnb_prob_age.csv', index=False, encoding='utf-8')

print('**********************************************************************************************')
print('SVM开始建模......')
# 训练集
x = tf_idf[:cv_train_num]
y = df_all['Age'][:cv_train_num]
# 测试集
x_test = tf_idf[cv_train_num:]
y_test = df_all['Age'][cv_train_num:]

num_age_class = len(pd.value_counts(df_all['Age']))
# print(num_Age_class)  # 3

stack = np.zeros((x.shape[0], num_age_class))
# print(stack.shape)
stack_te = np.zeros((x_test.shape[0], num_age_class))
# print(stack_te.shape)

score_va = 0  # 验证集得分
score_te = 0  # 测试集得分

# 交叉验证
for i, (train, dev) in enumerate(StratifiedKFold(n_splits=n, random_state=0).split(x, y)):
    clf = svm.LinearSVC(loss='hinge', tol=0.000001, C=0.5, random_state=seed, max_iter=1000)
    clf.fit(x[train], y[train])
    y_pred_dev = clf.decision_function(x[dev])
    y_pred_test = clf.decision_function(x_test)

    # 计算得分
    score_va += accuracy_score(y[dev], clf.predict(x[dev]))
    score_te += accuracy_score(y_test, clf.predict(x_test))

    # 得分累加
    stack[dev] += y_pred_dev
    stack_te += y_pred_test

# 计算平均得分
score_va /= n
score_te /= n
print('验证集年龄的准确率:%f' % score_va)
print('测试集年龄的准确率:%f' % score_te)

stack_te /= n  # 归一化
stack_all = np.vstack([stack, stack_te])
df_stack = pd.DataFrame(index=range(len(df_all)))
for i in range(stack_all.shape[1]):
    df_stack['tfidf_svm_{}'.format(i)] = stack_all[:, i]

df_stack.to_csv(feat_path + 'tfidf/svm_prob_age.csv', index=False, encoding='utf-8')
