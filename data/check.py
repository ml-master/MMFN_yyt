import numpy as np
import pandas as pd
import os
import cv2
import torch
from PIL import Image
import numpy as np

data = pd.read_csv("/home/yutao/MMFN/dataset/gossipcop.csv")


def check_image_exists(image_path):
    return os.path.exists(image_path)


def read_image(image_path):
    try:
        print(f"读取图片: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"无法读取图片：{image_path}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)  # 使用占位符图像
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
            img = img.transpose((2, 0, 1))  # 转换为 (C, H, W) 格式
    except Exception as e:
        print(f"读取图片时发生错误：{e}")
        img = np.zeros((224, 224, 3), dtype=np.uint8)  # 使用占位符图像
    return torch.tensor(img)


print(data.info())
# 检查data['has_top_img']是否都有图片
print(data['has_top_img'].value_counts())
data['image_exists'] = data['image'].apply(check_image_exists)

# print(data.info())
# print(data['image_exists'].value_counts())
# print(data['has_top_img'].value_counts())

print(os.getcwd())
test_image = '/home/yutao/MMFN/dataset/image/top_img/gossipcop-898419_top_img.png'
# print(check_image_exists(test_image))
# img = cv2.imread(test_image, cv2.IMREAD_COLOR)
# print(img.shape)

# img = img.astype(np.float32) / 255.
# image1 = read_image(test_image)
# image2 = read_image(test_image2)
# print(f"{test_image}: {os.access(test_image, os.R_OK)}")
# print("image1 shape:", image1.shape)
# print("image2 shape:", image2.shape)

image1 = Image.open(test_image)
im_np = np.array(image1)
# print(im_np.shape)
if im_np.shape[2] == 2:
    im_np = np.expand_dims(im_np, axis=2)
    # im_np = np.repeat(im_np, 3, axis=2)
# print(im_np.shape)
cvim = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)  #
print(cvim.shape)
# cv2.imshow('image', cvim)
