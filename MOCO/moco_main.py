import torch
import torch.optim as optim
from dataset import Dataset_forMOCO, data_augmentation
from config_parser import config_args
from torch.utils.data import DataLoader
from models import MoCoV2
from trainer import MoCo_Trainer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    # Empty cache
    torch.cuda.empty_cache()
    # Get config args
    dataset_args = config_args['dataset']
    moco_args = config_args['moco_model']
    transform_args = config_args['transform_augmentation']
    dataloader_args = config_args['dataloader']

    # Get augmentation transform
    aug_transform = data_augmentation(transform_args)

    # Get train dataset
    train_dataset = Dataset_forMOCO(dataset_args=dataset_args, val=False,
                                    transform_flag=True, aug_transform=aug_transform)

    # Get train dataloader
    train_dl = DataLoader(train_dataset,
                          batch_size=dataloader_args['batch_size'],
                          num_workers=dataloader_args['num_workers'],
                          shuffle=True,
                          pin_memory=True)

    # Get validation dataset
    val_dataset = Dataset_forMOCO(dataset_args=dataset_args, val=True,
                                  transform_flag=True, aug_transform=aug_transform)

    # Get validation dataloader
    val_dl = DataLoader(val_dataset,
                        batch_size=dataloader_args['batch_size'],
                        num_workers=dataloader_args['num_workers'],
                        shuffle=True,
                        pin_memory=True)

    # Initialize MoCo model
    moco_model = MoCoV2(moco_args)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize trainer
    trainer = MoCo_Trainer(moco_model,
                           moco_model_args=moco_args,
                           optimizer=optim.SGD(moco_model.f_q.parameters(),
                                               lr=moco_args['optim']['lr'],
                                               momentum=moco_args['optim']['momentum'],
                                               weight_decay=moco_args['optim']['weight_decay']),
                           device=device)

    # Train MoCo model
    trainer.fit(train_dl, val_dl)
