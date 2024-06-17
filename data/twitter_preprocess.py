import pandas
import pandas as pd
import os
from pandas import json_normalize
import json
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

pandas.set_option('display.max_columns', None)
path = '/home/yutao/MMFN/dataset/twitter_dataset'
# print path to check if it is correct
print(os.getcwd())
# Read the CSV file
train_data = pd.read_csv(path + '/train' + '/train_tweets.txt', sep='\t', header=0)
test_data = pd.read_csv(path + '/test' + '/test_tweets.txt', sep='\t', header=0)
# print(train_data.head(
#
# ))
print(test_data)
# 查询图片是否存在
image_train_path = path + '/train' + '/all_images'
image_test_path = path + '/test' + '/all_images'
# 将文件夹中所有图片放在一个文件夹中
# import shutil

# def move_images(source_dir, target_dir, image_extensions=None):
#     # 创建目标文件夹如果不存在
#     if image_extensions is None:
#         image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#
#     # 遍历源目录中的所有文件和子文件夹
#     for root, dirs, files in os.walk(source_dir):
#         for file in files:
#             if any(file.lower().endswith(ext) for ext in image_extensions):
#                 # 构建源文件的完整路径
#                 source_file_path = os.path.join(root, file)
#                 # 构建目标文件的完整路径
#                 target_file_path = os.path.join(target_dir, file)
#
#                 # 处理重名文件的情况
#                 if os.path.exists(target_file_path):
#                     base, ext = os.path.splitext(file)
#                     count = 1
#                     while os.path.exists(target_file_path):
#                         target_file_path = os.path.join(target_dir, f"{base}_{count}{ext}")
#                         count += 1
#
#                 # 移动文件
#                 shutil.move(source_file_path, target_file_path)
#                 print(f"Moved: {source_file_path} -> {target_file_path}")
#
#
# source_directory = path + '/test' + '/TestSetImages'  # 替换为你的源目录路径
# target_directory = path + '/test' + '/all_images'  # 替换为你的目标目录路径
#
# move_images(source_directory, target_directory)


train_data['has_image'] = 1
test_data['has_image'] = 1
train_data['is_video'] = 0
test_data['is_video'] = 0
video_names = ['syrianboy_1', 'varoufakis_1']


# 查询图片是否存在
def check_and_update_image(row, image_path):
    images_list = row['imageId(s)'].split(',')
    image_name = images_list[0].strip()  # 仅保留第一张图片
    jpg_path = os.path.join(image_path, image_name + '.jpg')
    png_path = os.path.join(image_path, image_name + '.png')
    if row['imageId(s)'] in video_names:
        row['has_image'] = 0
        row['is_video'] = 1
    if os.path.exists(jpg_path):
        row['imageId(s)'] = jpg_path
    elif os.path.exists(png_path):
        row['imageId(s)'] = png_path
    else:
        row['has_image'] = 0

    return row


def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False


# 检查并更新 train_data 和 test_data 的图像路径
train_data = train_data[train_data['tweetText'].apply(is_english)].apply(check_and_update_image, axis=1,
                                                                         image_path=image_train_path)
test_data = test_data[test_data['tweetText'].apply(is_english)].apply(check_and_update_image, axis=1,
                                                                      image_path=image_test_path)
train_data = train_data[train_data['has_image'] == 1]
test_data = test_data[test_data['has_image'] == 1]
train_data = train_data[train_data['is_video'] == 0]
test_data = test_data[test_data['is_video'] == 0]
train_data['label'] = train_data['label'].apply(lambda x: 0 if x == 'fake' else 1)
test_data['label'] = test_data['label'].apply(lambda x: 0 if x == 'fake' else 1)
# 保存处理后的数据
train_data.to_csv(os.path.join(path, 'train_tweets_preprocess.csv'), index=False)
test_data.to_csv(os.path.join(path, 'test_tweets_preprocess.csv'), index=False)

# train_data = pd.read_csv(os.path.join(path, 'train_tweets_preprocess.csv'))
# test_data = pd.read_csv(os.path.join(path, 'test_tweets_preprocess.csv'))

print(train_data['label'].value_counts())
print(test_data['label'].value_counts())
