from io import StringIO

import pandas
import pandas as pd
import os

pandas.set_option('display.max_columns', None)
from pandas import json_normalize
import json
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

pandas.set_option('display.max_columns', None)
path = '/home/yutao/MMFN/dataset/weibo'
# print path to check if it is correct
# print(os.getcwd())
# Read the CSV file
data_path = path

train_data = pd.read_csv(data_path + '/train_weibov.csv', header=0)
test_data = pd.read_csv(data_path + '/test_weibov.csv', header=0)
print(train_data.head())


# def read_tweet_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#
#     # Initialize lists to store data
#     meta_info_list = []
#     first_image_suffix_list = []
#     tweet_content_list = []
#
#     # Process every three lines as one tweet record
#     for i in range(0, len(lines), 3):
#         meta_info = lines[i].strip().split('|')
#         image_urls = lines[i + 1].strip().split('|')
#         tweet_content = lines[i + 2].strip()
#
#         # Extract the last part of the first image URL
#         if image_urls[0] != 'null':
#             first_image_url = image_urls[0]
#             first_image_suffix = first_image_url.split('/')[-1]  # Extract the last part of the URL
#         else:
#             first_image_suffix = 'null'
#
#         meta_info_list.append(meta_info)
#         first_image_suffix_list.append(first_image_suffix)
#         tweet_content_list.append(tweet_content)
#
#     # Convert lists to DataFrame
#     columns = [
#         'tweet_id', 'user_name', 'tweet_url', 'user_url', 'publish_time',
#         'original', 'retweet_count', 'comment_count', 'praise_count', 'user_id',
#         'user_auth_type', 'user_fans_count', 'user_follow_count', 'user_tweet_count',
#         'publish_platform', 'first_image_suffix', 'tweet_content'
#     ]
#
#     data = []
#     for meta, image_suffix, content in zip(meta_info_list, first_image_suffix_list, tweet_content_list):
#         # Merge metadata with first image suffix and tweet content
#         data.append(meta + [image_suffix, content])
#
#     df = pd.DataFrame(data, columns=columns)
#
#     return df
#
#
def check_and_update_image(row, image_path):
    image_name = image_path + '/' + row['images']
    row['images'] = image_name
    if os.path.exists(image_name):
        row['has_image'] = 1
    return row


train_data['has_image'] = 0
test_data['has_image'] = 0
train_data = train_data.apply(check_and_update_image, axis=1, image_path=path)
test_data = test_data.apply(check_and_update_image, axis=1, image_path=path)
print(train_data['has_image'].value_counts())
print(test_data['has_image'].value_counts())
train_data.to_csv(os.path.join(path, 'train_weibo_preprocess.csv'), index=False)
test_data.to_csv(os.path.join(path, 'test_weibo_preprocess.csv'), index=False)
#
#
# # Example usage
# train_rumor_df = read_tweet_file(data_path + 'train_rumor.txt')
# train_nonrumor_df = read_tweet_file(data_path + 'train_nonrumor.txt')
# test_rumor_df = read_tweet_file(data_path + 'test_rumor.txt')
# test_nonrumor_df = read_tweet_file(data_path + 'test_nonrumor.txt')
#
# # Print the head of one of the DataFrames to verify
# # 整合
# train_rumor_df['label'] = 1
# train_nonrumor_df['label'] = 0
# test_rumor_df['label'] = 1
# test_nonrumor_df['label'] = 0
# train_rumor_df['image_path'] = train_rumor_df['first_image_suffix'].apply(
#     lambda x: os.path.join(path + '/rumor_images/', x))
# train_nonrumor_df['image_path'] = train_nonrumor_df['first_image_suffix'].apply(
#     lambda x: os.path.join(path + '/nonrumor_images/', x))
# test_rumor_df['image_path'] = test_rumor_df['first_image_suffix'].apply(
#     lambda x: os.path.join(path + '/rumor_images/', x))
# test_nonrumor_df['image_path'] = test_nonrumor_df['first_image_suffix'].apply(
#     lambda x: os.path.join(path + '/nonrumor_images/', x))
#
# print(train_rumor_df['image_path'][0])
#
# train_data = pd.concat([train_rumor_df, train_nonrumor_df])
# train_data['has_image'] = 0
# test_data = pd.concat([test_rumor_df, test_nonrumor_df])
# test_data['has_image'] = 0
# train_data = train_data.apply(check_and_update_image, axis=1)
# test_data = test_data.apply(check_and_update_image, axis=1)
# print(train_data['has_image'].value_counts())
# print(test_data['has_image'].value_counts())
# # 查看没有图片的数据
# # 仅保留tweet_id，tweet_content,label,has_image,image_path列
# train_data = train_data[['tweet_id', 'tweet_content', 'label', 'has_image', 'image_path']]
# test_data = test_data[['tweet_id', 'tweet_content', 'label', 'has_image', 'image_path']]
# print(train_data.info())
