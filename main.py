from dataset import DatasetForVQA, create_df, load_data
from config_parser import config_args
from torch.utils.data import DataLoader
from model import QAModel
import torch
from transformers import logging
from train import QA_Trainer
from sklearn import preprocessing
import pandas as pd

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Remove annoying warnings
    logging.set_verbosity(50)

    # Parse configuration args
    dataset_args = config_args['dataset']
    QA_model_args = config_args['QA_model']

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

    # Create the different datasets
    trainset = DatasetForVQA(dataset_args=dataset_args, df=train_df)
    valset = DatasetForVQA(dataset_args=dataset_args, df=val_df)
    testset = DatasetForVQA(dataset_args=dataset_args, df=test_df)

    # Create the different dataloaders
    trainloader = DataLoader(trainset,
                             batch_size=QA_model_args['batch_size'],
                             num_workers=QA_model_args['num_workers'],
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True)

    valloader = DataLoader(valset,
                           batch_size=QA_model_args['batch_size'],
                           num_workers=QA_model_args['num_workers'],
                           shuffle=True,
                           pin_memory=True,
                           drop_last=True)

    testloader = DataLoader(testset,
                            batch_size=QA_model_args['batch_size'],
                            num_workers=QA_model_args['num_workers'],
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True)

    # Load model with appropriate args and load to device
    model = QAModel(model_args=QA_model_args, device=device)

    # Loads the state_dict from the best trained model. Comment if you want to train from the beginning
    # model.load_state_dict(torch.load('QA_Logs/QA_Classifier_55_73_1TransformerOnFeatMap/best_QA_model.pt'))

    # Initialize the trainer
    trainer = QA_Trainer(QA_model=model,
                         QA_model_args=QA_model_args,
                         optimizer=torch.optim.AdamW(model.parameters(), lr=5e-6),
                         device=device,
                         ref_arr=le.classes_)

    # Fit the model to the data using the trainer. Comment to skip the training
    trainer.fit(trainloader, valloader)

    # Evaluate model on test set
    loss, acc1, acc5, bleu = trainer.eval(dl_val=testloader)
    print('BLEU Score:', bleu)
    print('Top1 Accuracy:', acc1.item())
    print('Top5 Accuracy:', acc5)
    print("Done!")
