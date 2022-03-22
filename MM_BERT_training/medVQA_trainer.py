import imp
import os
import numpy as np
import pandas as pd
import random
import math
import json

import torch
from torchvision import transforms, models
from torch.cuda.amp import GradScaler
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from PIL import Image
from random import choice
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import top_k_accuracy_score

# Utils


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


def calculate_bleu_score(preds, targets, idx2ans):
    bleu_per_answer = np.asarray(
        [sentence_bleu([str(idx2ans[target]).split()], str(idx2ans[pred]).split(), weights=[1]) for pred, target in
         zip(preds, targets)])


    return np.mean(bleu_per_answer)


def train_one_epoch(loader, model, optimizer, criterion, device, scaler, args, train_df, idx2ans,epoch,  writer):
    model.train()
    train_loss = []

    PREDS = []
    TARGETS = []

    # bar = tqdm(loader, leave=False)
    total_top_5_acc = 0.0
    step = 0
    for (img, question_token, segment_ids, attention_mask, target) in loader:

        img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(
            device), segment_ids.to(device), attention_mask.to(device), target.to(device)
        question_token = question_token.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        loss_func = criterion
        optimizer.zero_grad()

        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                logits, _ = model(img, question_token, segment_ids, attention_mask)
                loss = loss_func(logits, target)
        else:
            logits, _ = model(img, question_token, segment_ids, attention_mask)
            loss = loss_func(logits, target)

        if args.mixed_precision:
            scaler.scale(loss)
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
        
        total_top_5_acc += top_k_accuracy_score(
            target.cpu().detach().numpy(), 
            logits.softmax(1).cpu().detach().numpy(),
            k=5,
            labels=range(args.num_classes)
        ) 
        step += 1
        
        pred = logits.softmax(1).argmax(1).detach()
        PREDS.append(pred)
        if args.smoothing:
            TARGETS.append(target.argmax(1))
        else:
            TARGETS.append(target)

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        # bar.set_description('train_loss: %.5f' % loss_np)
    writer.add_scalar(tag='Loss/train', scalar_value=np.mean(train_loss), global_step=epoch + 1)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    total_acc = (PREDS == TARGETS).mean() * 100.
    avg_top_5_acc = (total_top_5_acc / step)*100.
    writer.add_scalar(tag='Acc/train', scalar_value=total_acc, global_step=epoch + 1)
    writer.add_scalar(tag='Top_5_Acc/train', scalar_value=avg_top_5_acc, global_step=epoch + 1)

    closed_acc = (PREDS[train_df['answer_type'] == 'CLOSED'] == TARGETS[
        train_df['answer_type'] == 'CLOSED']).mean() * 100.
    open_acc = (PREDS[train_df['answer_type'] == 'OPEN'] == TARGETS[train_df['answer_type'] == 'OPEN']).mean() * 100.

    writer.add_scalar(tag='Closed_Acc/train', scalar_value=closed_acc, global_step=epoch + 1)
    writer.add_scalar(tag='Open_Acc/train', scalar_value=open_acc, global_step=epoch + 1)

    acc = {'total_acc': np.round(total_acc, 4), 'closed_acc': np.round(closed_acc, 4),
           'open_acc': np.round(open_acc, 4), 'top_5_acc': np.round(avg_top_5_acc, 4)}

    return np.mean(train_loss), acc


def validate(loader, model, criterion, device, scaler, args, val_df, idx2ans, epoch, writer):
    model.eval()
    val_loss = []

    PREDS = []
    TARGETS = []
    # bar = tqdm(loader, leave=False)
    total_top_5_acc = 0.0
    step = 0

    with torch.no_grad():
        for (img, question_token, segment_ids, attention_mask, target) in loader:

            img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(
                device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)

            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits, _ = model(img, question_token, segment_ids, attention_mask)
                    loss = criterion(logits, target)
            else:
                logits, _ = model(img, question_token, segment_ids, attention_mask)
                loss = criterion(logits, target)

            loss_np = loss.detach().cpu().numpy()

            pred = logits.softmax(1).argmax(1).detach()

            PREDS.append(pred)

            if args.smoothing:
                TARGETS.append(target.argmax(1))
            else:
                TARGETS.append(target)

            total_top_5_acc += top_k_accuracy_score(
            target.cpu().detach().numpy(), 
            logits.softmax(1).cpu().detach().numpy(),
            k=5,
            labels=range(args.num_classes)
            ) 
            step += 1
            val_loss.append(loss_np)

            # bar.set_description('val_loss: %.5f' % (loss_np))

        val_loss = np.mean(val_loss)
        writer.add_scalar(tag='Loss/val', scalar_value=val_loss, global_step=epoch + 1)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    total_acc = (PREDS == TARGETS).mean() * 100.
    avg_top_5_acc = (total_top_5_acc / step)*100.

    closed_acc = (PREDS[val_df['answer_type'] == 'CLOSED'] == TARGETS[val_df['answer_type'] == 'CLOSED']).mean() * 100.
    open_acc = (PREDS[val_df['answer_type'] == 'OPEN'] == TARGETS[val_df['answer_type'] == 'OPEN']).mean() * 100.

    writer.add_scalar(tag='Acc/val', scalar_value=total_acc, global_step=epoch + 1)
    writer.add_scalar(tag='Top_5_Acc/val', scalar_value=avg_top_5_acc, global_step=epoch + 1)

    writer.add_scalar(tag='Closed_Acc/val', scalar_value=closed_acc, global_step=epoch + 1)
    writer.add_scalar(tag='Open_Acc/val', scalar_value=open_acc, global_step=epoch + 1)

    acc = {'total_acc': np.round(total_acc, 4), 'closed_acc': np.round(closed_acc, 4),
           'open_acc': np.round(open_acc, 4), 'top_5_acc': np.round(avg_top_5_acc, 4)}

    # add bleu score code
    total_bleu = calculate_bleu_score(PREDS, TARGETS, idx2ans)
    closed_bleu = calculate_bleu_score(PREDS[val_df['answer_type'] == 'CLOSED'],
                                       TARGETS[val_df['answer_type'] == 'CLOSED'], idx2ans)
    open_bleu = calculate_bleu_score(PREDS[val_df['answer_type'] == 'OPEN'], TARGETS[val_df['answer_type'] == 'OPEN'],
                                     idx2ans)

    bleu = {'total_bleu': np.round(total_bleu, 4), 'closed_bleu': np.round(closed_bleu, 4),
            'open_bleu': np.round(open_bleu, 4)}
    writer.add_scalar(tag='blue/val', scalar_value=total_bleu, global_step=epoch + 1)
    writer.add_scalar(tag='Closed_blue/val', scalar_value=closed_bleu, global_step=epoch + 1)
    writer.add_scalar(tag='Open_blue/val', scalar_value=open_bleu, global_step=epoch + 1)
    return val_loss, PREDS, acc, bleu


def test(loader, model, criterion, device, scaler, args, val_df, idx2ans, epoch, writer):
    model.eval()

    PREDS = []
    TARGETS = []

    test_loss = []
    total_top_5_acc = 0.0
    step = 0
    with torch.no_grad():
        for (img, question_token, segment_ids, attention_mask, target) in loader:

            img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(
                device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)

            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits, _ = model(img, question_token, segment_ids, attention_mask)
                    loss = criterion(logits, target)
            else:
                logits, _ = model(img, question_token, segment_ids, attention_mask)
                loss = criterion(logits, target)

            loss_np = loss.detach().cpu().numpy()

            test_loss.append(loss_np)

            pred = logits.softmax(1).argmax(1).detach()

            PREDS.append(pred)

            if args.smoothing:
                TARGETS.append(target.argmax(1))
            else:
                TARGETS.append(target)
            
            total_top_5_acc += top_k_accuracy_score(
            target.cpu().detach().numpy(), 
            logits.softmax(1).cpu().detach().numpy(),
            k=5,
            labels=range(args.num_classes)
            ) 
            step += 1
        test_loss = np.mean(test_loss)
        writer.add_scalar(tag='Loss/test', scalar_value=test_loss, global_step=epoch + 1)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    total_acc = (PREDS == TARGETS).mean() * 100.
    avg_top_5_acc = (total_top_5_acc / step) *100.

    closed_acc = (PREDS[val_df['answer_type'] == 'CLOSED'] == TARGETS[val_df['answer_type'] == 'CLOSED']).mean() * 100.
    open_acc = (PREDS[val_df['answer_type'] == 'OPEN'] == TARGETS[val_df['answer_type'] == 'OPEN']).mean() * 100.

    writer.add_scalar(tag='Acc/test', scalar_value=total_acc, global_step=epoch + 1)
    writer.add_scalar(tag='Top_5_Acc/test', scalar_value=avg_top_5_acc, global_step=epoch + 1)

    writer.add_scalar(tag='Closed_Acc/test', scalar_value=closed_acc, global_step=epoch + 1)
    writer.add_scalar(tag='Open_Acc/test', scalar_value=open_acc, global_step=epoch + 1)

    acc = {'total_acc': np.round(total_acc, 4), 'closed_acc': np.round(closed_acc, 4),
           'open_acc': np.round(open_acc, 4), 'top_5_acc': np.round(avg_top_5_acc, 4)}

    return test_loss, PREDS, acc


def final_test(loader, all_models, device, args, val_df, idx2ans):
    PREDS = []

    with torch.no_grad():
        for (img, question_token, segment_ids, attention_mask, target) in tqdm(loader, leave=False):

            img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(
                device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)

            for i, model in enumerate(all_models):
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits, _ = model(img, question_token, segment_ids, attention_mask)
                else:
                    logits, _ = model(img, question_token, segment_ids, attention_mask)

                if i == 0:
                    pred = logits.detach().cpu().numpy() / len(all_models)
                else:
                    pred += logits.detach().cpu().numpy() / len(all_models)

            PREDS.append(pred)

    PREDS = np.concatenate(PREDS)

    return PREDS
