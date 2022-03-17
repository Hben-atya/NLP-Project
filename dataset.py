import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import json
import pandas as pd


def load_data(path):
    # Load json
    with open(path) as f:
        data = json.load(f)
    # Some of the data in the json is irrelevant. This for-loop gets the relevant data and puts it in a list
    list_data = []
    for i in range(len(data)):
        list_data.append([
            data[i]['image_name'],
            data[i]['question'],
            data[i]['answer'],
            data[i]['image_organ'],
            data[i]['question_type'].split(', ')[0],
            data[i]['answer_type'].split(', ')[0]
        ])
    # Create a dataframe from the aforementioned list
    df = pd.DataFrame(list_data, columns=['Im_ID', 'Q', 'A', 'Im_Organ', 'Q_Type', 'A_Type'])
    # Some of the data is misspelled. This fixes it.
    df = df.replace(['ATRIB', 'PRSE', 'Other', 'CLOSED '],
                    ['ATTRIB', 'PRES', 'OTHER', 'CLOSED'])
    # Replace the strings for the question, answer, and image classes with unique integers
    df = df.replace(['PRES', 'ABN', 'MODALITY', 'ORGAN', 'POS', 'PLANE', 'OTHER', 'SIZE', 'COUNT', 'ATTRIB', 'COLOR',
                     'CLOSED', 'OPEN',
                     'HEAD', 'CHEST', 'ABD'],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     0, 1,
                     0, 1, 2])
    return df


def create_df(path, le):
    # Load the dataframe using the load_data function
    df = load_data(path)

    # Reorganize the answers
    df['A'] = pd.Series(str(x).split('\t')[0].lower() for x in list(df['A']))

    # Use the label encoder to change answer strings to unique integers
    df['A_Labels'] = le.transform(list(df['A']))

    return df


def logits_to_str(logits, A_Q_Im='Im'):
    # From logits, get the predicted class (argmax)
    cls = torch.argmax(logits, dim=1)

    # Replace all of the unique integer classes with their original strings
    str_cls = []
    for i in range(len(cls)):

        if A_Q_Im == 'Im':  # Image classes
            if cls[i] == 0:
                str_cls.append('HEAD')
            elif cls[i] == 1:
                str_cls.append('CHEST')
            elif cls[i] == 2:
                str_cls.append('ABD')

        if A_Q_Im == 'A':  # Answer classes
            if cls[i] == 0:
                str_cls.append('CLOSED')
            elif cls[i] == 1:
                str_cls.append('OPEN')

        if A_Q_Im == 'Q':  # Question classes
            if cls[i] == 0:
                str_cls.append('PRES')
            elif cls[i] == 1:
                str_cls.append('ABN')
            elif cls[i] == 2:
                str_cls.append('MODALITY')
            elif cls[i] == 3:
                str_cls.append('ORGAN')
            elif cls[i] == 4:
                str_cls.append('POS')
            elif cls[i] == 5:
                str_cls.append('PLANE')
            elif cls[i] == 6:
                str_cls.append('OTHER')
            elif cls[i] == 7:
                str_cls.append('SIZE')
            elif cls[i] == 8:
                str_cls.append('COUNT')
            elif cls[i] == 9:
                str_cls.append('ATTRIB')
            elif cls[i] == 10:
                str_cls.append('COLOR')

    return str_cls


class DatasetForVQA(Dataset):
    # Dataset for VQA and Image Classification depending on the im_cls flag
    def __init__(self, dataset_args, df, im_cls=False):
        self.data_path = dataset_args['im_path']  # For get_item
        self.df = df  # data
        self.resize = transforms.Resize((528, 528))  # Resize transform for image
        self.pil2tensor = transforms.PILToTensor()  # Pillow to tensor transform

        # Image classifier flag. If True, the dataset returned is for image classification,
        # else returns the dataset for QA.
        self.im_cls = im_cls

    def __len__(self):
        return self.df.shape[0]  # Data length

    def __getitem__(self, item):
        # Path to image
        img_path = os.path.join(self.data_path, self.df['Im_ID'][item])

        # Get image
        img = Image.open(img_path).convert("L")

        # Pil to Tensor & transforms
        img_tensor = self.pil2tensor(img).float()
        img_tensor = self.resize(img_tensor)
        img_tensor = img_tensor.repeat(3, 1, 1)
        data = {
            'Q': self.df['Q'][item],  # Question
            'A': self.df['A'][item],  # Answer
            'A_Labels': self.df['A_Labels'][item],  # Answer after integer labeling
            'Im_Tensor': img_tensor  # Image tensor
        }
        if self.im_cls:
            data = (
                img_tensor,  # Image tensor
                self.df['Im_Organ'][item]  # Image class
            )

        return data


class QA_Type_Dataset(Dataset):
    # Dataset for question and answer classification
    def __init__(self, encodings, labels):
        self.encodings = encodings  # Encodings
        self.labels = labels  # Labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}  # Question or answer encodings
        item['labels'] = torch.tensor(self.labels[idx])  # label tensor
        return item

    def __len__(self):
        return len(self.labels)  # Data length
