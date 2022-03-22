from model import ClsModel
import torch
from pytorch_lightning import Trainer
from dataset import DatasetForVQA, create_df, load_data
from config_parser import config_args
from torch.utils.data import DataLoader
from sklearn import preprocessing
import pandas as pd

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Parse configuration args
    dataset_args = config_args['dataset']
    Im_cls_model_args = config_args['Im_Cls_model']\

    # Load data
    df = load_data(r"C:\Users\dekel\OneDrive\Masters\NLP\NLP_Course_Project\VQA_RAD_DATA\VQA_RAD Dataset Public.json")
    # Reorganize answers
    df['A'] = pd.Series(str(x).split('\t')[0].lower() for x in list(df['A']))
    # Make a label encoder to map answers to unique integers
    le = preprocessing.LabelEncoder()
    le.fit(list(df['A']))

    # Create the different dataframes
    train_df = create_df('VQA_RAD_DATA/train_vqa_rad.json', le)
    val_df = create_df('VQA_RAD_DATA/val_vqa_rad.json', le)
    test_df = create_df('VQA_RAD_DATA/test_vqa_rad.json', le)

    # Initialize image classification model
    model = ClsModel()

    # Loads the state_dict from the best trained model. Comment if you want to train from the beginning
    checkpoint = torch.load("QA_Logs/Classifier_Logs/checkpoints/epoch=8-step=512.ckpt")
    model.load_state_dict((checkpoint['state_dict']))

    # Create different datasets
    trainset = DatasetForVQA(dataset_args=dataset_args, df=train_df, im_cls=True)
    valset = DatasetForVQA(dataset_args=dataset_args, df=val_df, im_cls=True)
    testset = DatasetForVQA(dataset_args=dataset_args, df=test_df, im_cls=True)

    # Create different dataloaders
    trainloader = DataLoader(trainset,
                             batch_size=Im_cls_model_args['batch_size'],
                             num_workers=Im_cls_model_args['num_workers'],
                             shuffle=True,
                             pin_memory=True)

    valloader = DataLoader(valset,
                           batch_size=Im_cls_model_args['batch_size'],
                           num_workers=Im_cls_model_args['num_workers'],
                           shuffle=False,
                           pin_memory=True)

    # Initialize pytorch-lightning trainer
    trainer = Trainer(
        gpus=1,
        precision=16
    )

    # Fit model to data using the trainer
    # trainer.fit(model=model,
    #             train_dataloader=trainloader,
    #             val_dataloaders=valloader)

    trainer.test(model=model, dataloaders=valloader)
