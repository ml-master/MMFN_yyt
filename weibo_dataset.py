import torch
import torch.utils.data as data
import data.util as util
import torchvision.transforms.functional as F
import pandas
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, AutoFeatureExtractor
import clip

# Determine whether to use CUDA (GPU) or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and its preprocessing function
clipmodel, preprocess = clip.load('ViT-B/32', device)

# Freeze the parameters of the CLIP model
for param in clipmodel.parameters():
    param.requires_grad = False

# Load a feature extractor from the transformers library
feature_extractor = AutoFeatureExtractor.from_pretrained("./swin-base-patch4-window7-224")


# Function to read images given their paths
def read_img(imgs, root_path, LABLEF):
    # Select a random image path from the provided list
    GT_path = imgs[np.random.randint(0, len(imgs))]
    if '/' in GT_path:
        GT_path = GT_path[GT_path.rfind('/') + 1:]
    GT_path = "{}/{}/{}".format(root_path, LABLEF, GT_path)
    # Read the Ground Truth (GT) image
    img_GT = util.read_img(GT_path)
    img_pro = Image.open(GT_path).convert('RGB')
    return img_GT, img_pro


# Custom dataset class
class weibo_dataset(data.Dataset):
    def __init__(self, root_path='/data/ymzhou/weibo', is_train=True):
        super(weibo_dataset, self).__init__()
        self.is_train = is_train
        self.root_path = root_path
        self.index = 0
        self.label_dict = []
        self.swin = feature_extractor
        self.preprocess = preprocess
        self.local_path = '/data/ymzhou/dataset'

        # Read CSV file to populate label_dict
        wb = pandas.read_csv(self.local_path + '/{}_weibov.csv'.format('train' if is_train else 'test'))

        # Populate label_dict with records from the CSV file
        for i in tqdm(range(len(wb))):
            images_name = str(wb.iloc[i, 1]).lower()
            label = int(wb.iloc[i, 2])
            content = str(wb.iloc[i, 3]) + str(wb.iloc[i, 5])
            sum_content = str(wb.iloc[i, 4])
            record = {}
            record['images'] = images_name
            record['label'] = label
            record['content'] = content
            record['sum_content'] = sum_content
            self.label_dict.append(record)
        assert len(self.label_dict) != 0, 'Error: GT path is empty.'

    def __getitem__(self, index):

        # get GT image
        record = self.label_dict[index]
        images, label, content, sum_content = record['images'], record['label'], record['content'], record[
            'sum_content']
        if label == 0:
            LABLEF = 'rumor_images'
        else:
            LABLEF = 'nonrumor_images'
        imgs = images.split('|')
        try:
            img_GT, img_pro = read_img(imgs, self.root_path, LABLEF)
        except Exception:
            raise IOError("Load {} Error {}".format(imgs, record['images']))

        return (
            content, self.swin(img_GT, return_tensors="pt").pixel_values, self.preprocess(img_pro), sum_content), label

    def __len__(self):
        return len(self.label_dict)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t


# Populate label_dict with records from the CSV file
token = BertTokenizer.from_pretrained('bert-base-chinese')


# Populate label_dict with records from the CSV file
def collate_fn(data):
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    imageclip = [i[0][2] for i in data]
    textclip = [i[0][3] for i in data]
    labels = [i[1] for i in data]

    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=300,
                                   return_tensors='pt',
                                   return_length=True)

    textclip = clip.tokenize(textclip, truncate=True)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    image = torch.stack(image).squeeze()
    imageclip = torch.stack(imageclip)
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, image, imageclip, textclip, labels
