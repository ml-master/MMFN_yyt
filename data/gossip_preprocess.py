import pandas
import pandas as pd
import os
from pandas import json_normalize
import json

pandas.set_option('display.max_columns', None)
path = '/home/yutao/MMFN/dataset'
# print path to check if it is correct
print(os.getcwd())
# Read the CSV file
image_path = path + '/image/top_img/'
with open(path + '/gossipcop_v3-1_style_based_fake.json', 'r') as file:
    data = json.load(file)
fake_data = json_normalize(data.values())
fake_data['label'] = fake_data['generated_label'].apply(lambda x: 1 if x == 'real' else 0)
fake_data['image'] = fake_data[['origin_id']].apply(lambda x: image_path + x[0] + '_top_img' + '.png', axis=1)
print(fake_data['has_top_img'].value_counts())
print(fake_data.info())
# 合并两个 DataFrame
data = fake_data
data.to_csv(path + '/gossipcop.csv', index=False)
# 划分数据集
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=42)
train.to_csv(path + '/train_gossipcop.csv', index=False)
test.to_csv(path + '/test_gossipcop.csv', index=False)
print(train.info())
print(test.info())
