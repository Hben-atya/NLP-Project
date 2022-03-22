import imp
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
from transformers import BertTokenizer, BertModel
import json

class Dataset_forMOCO(Dataset):

    def __init__(self, dataset_args, transform_flag=True, aug_transform=None, val=False):
        self.data_path = dataset_args['data_path_train']  # For get_item
        if val:
            self.data_path = dataset_args['data_path_val']
        self.aug_transform = aug_transform  # Transform function for images
        self.transform_flag = transform_flag  # Technically should always be True

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, item):
        # Path to image
        img_path = os.path.join(self.data_path, os.listdir(self.data_path)[item])
        # Get image
        img = Image.open(img_path).convert("L")
        # Apply transformation if exists (again, should technically exist)
        if self.transform_flag:
            # Resize image to (300,300) before augmenting for computational reasons
            resize = transforms.Resize((300, 300))
            # Pil to Tensor
            pil2tensor = transforms.PILToTensor()
            img_tensor = pil2tensor(img).float()
            img_tensor = resize(img_tensor)
            # If image is grayscale, make 3-channel grayscale where channel 1=2=3
            # if img_tensor.shape[0] == 1:
            #     img_tensor = img_tensor.repeat(3, 1, 1)
            # Apply random transformation twice, creating 2 different transforms of same image
            img1 = self.aug_transform(img_tensor)
            img2 = self.aug_transform(img_tensor)
        # Create pair of the two images
        data = {'image1': img1, 'image2': img2}
        return data


def data_augmentation(transform_args):
    """
    Transformations as described in MoCo original paper [Technical details section]
    :param transform_args:
    :return: aug_transform:
    """
    gaussian_blur_ = transform_args['GaussianBlur']  # Gaussian blur args
    # Sequential of: RandomResizedCrop, Normalization, RandomApply of
    # Gaussian Blur,Random Horizontal Flip with appropriate arguments
    aug_transform = nn.Sequential(
        transforms.RandomResizedCrop(transform_args['SizeCrop'], scale=(0.25, 1.0)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomApply(torch.nn.ModuleList(
            [transforms.GaussianBlur(kernel_size=gaussian_blur_['kernel_size'],
                                     sigma=(gaussian_blur_['sigma_start'], gaussian_blur_['sigma_end']))
             ]), p=transform_args['p_apply']),

        transforms.RandomHorizontalFlip(p=transform_args['RandomHorizontalFlip']),
    )

    return aug_transform


def load_data(args):
    train_file = os.path.join(args.data_dir, 'VQAnswering_2020_Train_QA_pairs.txt')
    val_file = os.path.join(args.data_dir, 'VQAnswering_2020_Val_QA_Pairs.txt')
    test_file = os.path.join(args.data_dir, 'VQAnswering_2020_Test_Questions.txt')

    train_df = pd.read_csv(train_file, sep='|', names=['img_id', 'question', 'answer'])
    train_df['mode'] = 'train'

    val_df = pd.read_csv(val_file, sep='|', names=['img_id', 'question', 'answer'])
    val_df['mode'] = 'val'

    test_df = pd.read_csv(test_file, sep='|', names=['img_id', 'question', 'answer'])
    test_df['mode'] = 'test'

    train_df['image_name'] = train_df['img_id'].apply(lambda x: os.path.join(args.data_dir, r'Images/train', x + '.jpg'))
    val_df['image_name'] = val_df['img_id'].apply(lambda x: os.path.join(args.data_dir, r'Images/val', x + '.jpg'))
    test_df['image_name'] = test_df['img_id'].apply(lambda x: os.path.join(args.data_dir, r'Images/test', x + '.jpg'))

    # train_df['question_type'] = train_df['question_type'].str.lower()
    # test_df['question_type'] = test_df['question_type'].str.lower()

    return train_df, val_df, test_df


def load_vqarad_data(train_json, val_json, test_json, data_dir):
    
    train_data = json.load(open(train_json),)
    val_data = json.load(open(val_json),)
    test_data = json.load(open(test_json),)

    traindf = pd.DataFrame(train_data) 
    traindf['mode'] = 'train'
    
    valdf = pd.DataFrame(val_data) 
    valdf['mode'] = 'val'
    
    testdf = pd.DataFrame(test_data)
    testdf['mode'] = 'test' 

    traindf['image_name'] = traindf['image_name'].apply(lambda x: os.path.join(data_dir, 'VQA_RAD_Image_Folder', x))
    valdf['image_name'] = valdf['image_name'].apply(lambda x: os.path.join(data_dir, 'VQA_RAD_Image_Folder', x))
    testdf['image_name'] = testdf['image_name'].apply(lambda x: os.path.join(data_dir, 'VQA_RAD_Image_Folder', x))

    traindf['question_type'] = traindf['question_type'].str.lower()
    valdf['question_type'] = valdf['question_type'].str.lower()
    testdf['question_type'] = testdf['question_type'].str.lower()

    return traindf, valdf, testdf


def encode_text(caption, tokenizer, args):
    part1 = [0 for _ in range(5)]
    # get token ids and remove [CLS] and [SEP] token id
    part2 = tokenizer.encode(caption)[1:-1]

    tokens = [tokenizer.cls_token_id] + part1 + [tokenizer.sep_token_id] + part2[:args.max_position_embeddings - 8] + [
        tokenizer.sep_token_id]
    segment_ids = [0] * (len(part1) + 2) + [1] * (len(part2[:args.max_position_embeddings - 8]) + 1)
    input_mask = [1] * len(tokens)
    n_pad = args.max_position_embeddings - len(tokens)
    tokens.extend([0] * n_pad)
    segment_ids.extend([0] * n_pad)
    input_mask.extend([0] * n_pad)

    return tokens, segment_ids, input_mask


class VQA_Dataset(Dataset):
    def __init__(self, df, tfm, args, mode='train'):
        self.df = df.values
        self.tfm = tfm
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        im_path = self.df[idx, 4]
        question = self.df[idx, 7]
        answer = self.df[idx, 12]

        # upload the image
        img = Image.open(im_path)

        # apply transform
        if self.tfm:
            img = self.tfm(img)

        tokens, segment_ids, input_mask = encode_text(question, self.tokenizer, self.args)

        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        segment_ids_tensor = torch.tensor(segment_ids, dtype=torch.long)
        input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)
        answer_tensor = torch.tensor(answer, dtype=torch.long)

        if self.mode != 'train':
            tok_ques = self.tokenizer.tokenize(question)
            # return img, tokens_tensor, segment_ids_tensor, input_mask_tensor, answer_tensor, tok_ques
            return img, tokens_tensor, segment_ids_tensor, input_mask_tensor, answer_tensor

        else:
            return img, tokens_tensor, segment_ids_tensor, input_mask_tensor, answer_tensor
