train_data = './data/train.csv'
test_data = './data/test.csv'

stopwords_file = './data/stop_words.txt'

processed_data = './processed_data/'
processed_data_file = './processed_data/all_data.csv'

feat_path = './feat/'
output = './output/'
output_age_file = './output/age_result.csv'
output_gender_file = './output/gender_result.csv'
output_education_file = './output/education_result.csv'

model_path = './model/'
model_file = './model/w2v.model'

# 默认None
train_num = 10000
test_num = 8000


cv_train_num = 9000
n = 5 # 交叉验证的次数


w2v_dim = 300  # 词向量的维度
seed = 2019
