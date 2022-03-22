import argparse
from medVQA_Model import MedVQA_Model
from dataset import load_data, load_vqarad_data, VQA_Dataset
from medVQA_trainer import train_one_epoch, validate, test, LabelSmoothing
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, models
from torch.cuda.amp import GradScaler
from torchtoolbox.transform import Cutout
import os
import yaml
# from models import MoCoV2
from torch.utils.tensorboard import SummaryWriter
import logging


def get_args():
    parser = argparse.ArgumentParser(description="Finetune on VQARAD")
    parser.add_argument('--run_name', type=str, required=False, default='vqa_rad_resnet_bert')
    parser.add_argument('--data_dir', type=str, required=False,
                        default=r"/data/data_hadas/vqa_rad/",
                        help="path for data")
    parser.add_argument('--model_dir', type=str, required=False,
                        default=r'../val_loss_3.pt',
                        help="path to load weights")
    parser.add_argument('--moco_model_dir', type=str, required=False,
                        default=r'../MOCO/MOCO_Logs_500_Epochs',
                        help="path to load weights")
    parser.add_argument('--moco_config', type=str, required=False,
                        default=r'../MOCO/config_moco.yaml')
    parser.add_argument('--save_dir', type=str, required=False,
                        default=r"/data/data_hadas/vqa_rad/medVQA_models",
                        help="path to save weights")

    parser.add_argument('--use_pretrained', action='store_true', default=False, help="use pretrained weights or not")
    parser.add_argument('--mixed_precision', action='store_true', default=False, help="use mixed precision or not")
    parser.add_argument('--clip', action='store_true', default=False, help="clip the gradients or not")

    parser.add_argument('--seed', type=int, required=False, default=42, help="set seed for reproducibility")
    parser.add_argument('--num_workers', type=int, required=False, default=4, help="number of workers")
    parser.add_argument('--epochs', type=int, required=False, default=100, help="num epochs to train")
    parser.add_argument('--train_pct', type=float, required=False, default=1.0,
                        help="fraction of train samples to select")
    parser.add_argument('--valid_pct', type=float, required=False, default=1.0,
                        help="fraction of validation samples to select")
    parser.add_argument('--test_pct', type=float, required=False, default=1.0,
                        help="fraction of test samples to select")

    parser.add_argument('--max_position_embeddings', type=int, required=False, default=28,
                        help="max length of sequence")
    parser.add_argument('--batch_size', type=int, required=False, default=32, help="batch size")

    parser.add_argument('--lr', type=float, required=False, default=1e-4, help="learning rate'")
    parser.add_argument('--factor', type=float, required=False, default=0.1, help="factor for rlp")
    parser.add_argument('--patience', type=int, required=False, default=10, help="patience for rlp")
    parser.add_argument('--hidden_dropout_prob', type=float, required=False, default=0.3,
                        help="hidden dropout probability")
    parser.add_argument('--smoothing', type=float, required=False, default=None, help="label smoothing")

    # parser.add_argument('--image_size', type=int, required=False, default=300, help="image size")
    parser.add_argument('--image_size', type=int, required=False, default=224, help="image size")
    parser.add_argument('--hidden_size', type=int, required=False, default=768, help="hidden size")
    parser.add_argument('--vocab_size', type=int, required=False, default=30522, help="vocab size")
    parser.add_argument('--type_vocab_size', type=int, required=False, default=2, help="type vocab size")
    parser.add_argument('--heads', type=int, required=False, default=12, help="heads")
    parser.add_argument('--n_layers', type=int, required=False, default=4, help="num of layers")

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()
    args.run_name = 'vqa_rad_resnet_bert'

    train_json = os.path.join(args.data_dir, 'train_vqa_rad.json')
    val_json = os.path.join(args.data_dir, 'val_vqa_rad.json')
    test_json = os.path.join(args.data_dir, 'test_vqa_rad.json')

    train_df, val_df, test_df = load_vqarad_data(train_json, val_json, test_json, args.data_dir)

    df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    df['answer'] = df['answer'].str.lower()
    ans2idx = {ans: idx for idx, ans in enumerate(df['answer'].unique())}
    idx2ans = {idx: ans for ans, idx in ans2idx.items()}
    df['answer'] = df['answer'].map(ans2idx).astype(int)
    # df['answer_type'] = df['answer']
    # ans2type = {ans: 'CLOSED' if (ans == 'yes' or ans == 'no') else 'OPEN' for idx, ans in
    #             enumerate(df['answer'].unique())}
    # ans2idx = {ans: idx for idx, ans in enumerate(df['answer'].unique())}
    # idx2ans = {idx: ans for ans, idx in ans2idx.items()}
    # df['answer'] = df['answer'].map(ans2idx).astype(int)
    # df['answer_type'] = df['answer_type'].map(ans2type)

    train_df = df[df['mode'] == 'train'].reset_index(drop=True)
    val_df = df[df['mode'] == 'val'].reset_index(drop=True)
    test_df = df[df['mode'] == 'test'].reset_index(drop=True)

    num_classes = len(ans2idx)

    args.num_classes = num_classes

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.image_model = models.resnet152(pretrained=True)

    run_dir = os.path.join(args.save_dir, args.run_name)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    
    # get logger:
    log_name = os.path.join(run_dir, 'logs.log')
    logging.basicConfig(filename=log_name,
                        filemode='a',
                        format='%(asctime)s - d%(levelname)s:  %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger(log_name)

    logger.info('start train')

    model = MedVQA_Model(args)

    if args.use_pretrained:
        model.load_state_dict(torch.load(args.model_dir))

    model.classifier[2] = nn.Linear(args.hidden_size, num_classes)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=args.factor, verbose=True)

    if args.smoothing:
        criterion = LabelSmoothing(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()

    train_tfm = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.RandomResizedCrop(args.image_size, scale=(0.5, 1.0),
                                                                 ratio=(0.75, 1.333)),
                                    transforms.RandomRotation(10),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_tfm = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_ds = VQA_Dataset(train_df, tfm=train_tfm, args=args, mode='train')
    val_ds = VQA_Dataset(val_df, tfm=test_tfm, args=args, mode='val')
    test_ds = VQA_Dataset(test_df, tfm=test_tfm, args=args, mode='test')

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    val_best_acc = 0
    test_best_acc = 0
    best_loss = np.inf
    counter = 0

    writer = SummaryWriter(run_dir)
    for epoch in range(args.epochs):

        print(f'Epoch {epoch + 1}/{args.epochs}')

        train_loss, train_acc = train_one_epoch(train_dl, model, optimizer, criterion, device, scaler, args,
                                                train_df, idx2ans, epoch, writer)
        val_loss, val_predictions, val_acc, val_bleu = validate(val_dl, model, criterion, device, scaler, args,
                                                                val_df, idx2ans, epoch, writer)
        test_loss, test_predictions, test_acc = test(test_dl, model, criterion, device, scaler, args, test_df,
                                                     idx2ans, epoch, writer)

        scheduler.step(train_loss)

        log_dict = val_acc

        for k, v in val_acc.items():
            log_dict[k] = v

        log_dict['train_loss'] = train_loss
        log_dict['test_loss'] = val_acc
        log_dict['learning_rate'] = optimizer.param_groups[0]["lr"]

        # wandb.log(log_dict)

        content = f'Learning rate: {(optimizer.param_groups[0]["lr"]):.7f}, ' \
                  f'Train loss: {train_loss :.4f}, Train acc: {train_acc}, \n' \
                  f'val loss: {val_loss :.4f},  val acc: {val_acc}, val_bleu: {val_bleu}, \n ' \
                  f'test loss: {test_loss: .4f}, test acc: {test_acc}'
        print(content)

        if val_acc['total_acc'] > test_best_acc:
            torch.save(model.state_dict(), os.path.join(run_dir, f'{args.run_name}_test_acc.pt'))
            test_best_acc = val_acc['total_acc']
            print('save best model')
            logger.info(content)
            logger.info(f'saved best model in epoch {epoch+1}')

        if epoch + 1 % 10 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': optimizer.param_groups[0]["lr"],
                'test_best_acc': val_acc['total_acc']

            }
            torch.save(checkpoint, os.path.join(run_dir, f'{args.run_name}_checkpoints.pt'))
