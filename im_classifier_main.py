from model import ClsModel
import torch
from pytorch_lightning import Trainer
from dataset import DatasetForVQA, create_df
from config_parser import config_args
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Parse configuration args
    dataset_args = config_args['dataset']
    Im_cls_model_args = config_args['Im_Cls_model']\

    # Create different dataframes
    train_df = create_df('VQA_RAD_DATA/train_vqa_rad.json')
    val_df = create_df('VQA_RAD_DATA/val_vqa_rad.json')
    test_df = create_df('VQA_RAD_DATA/test_vqa_rad.json')

    # Initialize image classification model
    model = ClsModel()

    # Create different datasets
    trainset = DatasetForVQA(dataset_args=dataset_args, df=train_df, im_cls=True)
    valset = DatasetForVQA(dataset_args=dataset_args, df=val_df, im_cls=True)

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
    trainer.fit(model=model,
                train_dataloader=trainloader,
                val_dataloaders=valloader)
