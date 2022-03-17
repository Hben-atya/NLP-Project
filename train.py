import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
from sklearn.metrics import top_k_accuracy_score
from transformers.optimization import get_linear_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu


class QA_Trainer:
    def __init__(self, QA_model, QA_model_args, optimizer=None, device=None):
        self.QA_model = QA_model  # Question-Answering Model
        self.optimizer = optimizer  # Optimizer
        self.device = device  # Device
        self.QA_model_args = QA_model_args  # QA model arguments
        self.QA_logs_path = QA_model_args['log_path']  # QA model logging path
        self.writer = SummaryWriter(self.QA_logs_path)  # Summary writer (tensorboard) for logging

        # Send model to device
        if self.device:
            self.QA_model.to(device)

    def fit(self,
            dl_train: DataLoader,
            dl_dev: DataLoader
            ):
        scaler = torch.cuda.amp.GradScaler()  # Scaler for mixed precision (16/32 bit) computation
        # Set model to training mode
        self.QA_model.train()
        num_epochs = self.QA_model_args['num_epochs']  # Number of epochs
        val_step = self.QA_model_args['val_step']  # Validate every val_step steps
        save_every = self.QA_model_args['save_every']  # Save every save_every steps
        best_val_loss = 1e8  # Initialize best validation loss
        best_val_acc = 0  # Initialize best validation accuracy
        last_epoch = 0  # Initialize the last epoch (for train resuming purposes)

        # If resume_run argument is True, load last checkpoint and continue training from there
        if self.QA_model_args['resume_run'] and os.path.exists(os.path.join(self.QA_logs_path, 'checkpoint.pt')):
            checkpoints = torch.load(os.path.join(self.QA_logs_path, 'checkpoint.pt'))
            self.QA_model = checkpoints['QA_model']
            self.optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
            best_val_loss = checkpoints['best_val_loss']
            last_epoch = checkpoints['last_epoch']
        gamma = 0.5  # Gamma value for learning rate scheduler
        # Initialize learning rate scheduler (linear scheduler)
        scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=num_epochs)

        # Training loop
        for epoch in range(last_epoch, num_epochs):
            # Set model to training mode
            self.QA_model.train()
            startTime = time.time()  # Start time
            train_epoch_loss = 0  # Initialize epoch loss
            train_epoch_acc = 0  # Initialize epoch accuracy
            train_epoch_top5_acc = 0  # Initialize epoch top5 accuracy
            # Get data batches
            for batch_data in dl_train:
                Q = batch_data['Q']  # Question
                A = batch_data['A']  # Answer
                A_Labels = batch_data['A_Labels']  # Answer labels
                A_Labels = torch.LongTensor(A_Labels.type(torch.int64)).to(self.device)  # Cast to long
                Im = batch_data['Im_Tensor'].to(self.device)  # Send image to device

                # Get logits, labels and then loss
                with torch.cuda.amp.autocast():
                    outputs = self.QA_model(Q, A, Im, A_Labels)
                    loss = outputs[0]

                # Updates
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()


                # Get logits from model
                logits = outputs[1]

                # Get predictions
                pred = torch.argmax(logits, dim=1)

                # Update loss and metrics
                train_epoch_loss += float(loss)
                train_epoch_acc += torch.mean((pred == A_Labels).type(torch.float))

                # For top5 accuracy metric
                y_score = logits.cpu().detach().numpy()
                y_true = A_Labels.cpu().detach().numpy()
                train_epoch_top5_acc += top_k_accuracy_score(y_true, y_score, k=5, labels=range(517))

                # Reset loss to None to save memory
                loss = None

            # Update learning rate scheduler
            scheduler.step()

            endTime = time.time()  # End time
            torch.cuda.empty_cache()  # Empty cache to save memory

            avg_train_loss = train_epoch_loss / len(dl_train)  # Get average training loss
            avg_train_acc = train_epoch_acc / len(dl_train)  # Get average training accuracy
            avg_train_top5_acc = train_epoch_top5_acc / len(dl_train)  # Get average training top5 accuracy

            # Print statistics
            print('epoch {}, loss {}, acc {}, top5_acc {}, epoch time {} seconds, time remaining {} hours'.format(
                epoch + 1,
                avg_train_loss,
                avg_train_acc,
                avg_train_top5_acc,
                round(endTime - startTime, 2),
                round(num_epochs * (endTime - startTime) / 3600 - epoch * (endTime - startTime) / 3600, 2)
            ))

            # Log, Save, and Evaluate
            self.writer.add_scalar(tag='Loss/train_loss', scalar_value=avg_train_loss, global_step=epoch + 1)

            if (epoch + 1) % val_step == 0:
                avg_val_loss, avg_val_acc, avg_val_top5_acc = self.eval(dl_dev)
                self.writer.add_scalar(tag='Loss/val_loss', scalar_value=avg_val_loss, global_step=epoch + 1)
                self.writer.add_scalar(tag='Acc/val_acc', scalar_value=avg_val_acc, global_step=epoch + 1)
                self.writer.add_scalar(tag='Acc/val_top5_acc', scalar_value=avg_val_top5_acc, global_step=epoch + 1)
                print('epoch {}, val loss {}, val acc {}, val_top5_acc {}'.format(epoch + 1,
                                                                                  avg_val_loss,
                                                                                  avg_val_acc,
                                                                                  avg_val_top5_acc))
                if avg_val_acc > best_val_acc:
                    print('save new best model')
                    torch.save(self.QA_model.state_dict(), os.path.join(self.QA_logs_path, 'best_QA_model.pt'))
                    best_val_loss = avg_val_loss

            if (epoch + 1) % save_every == 0:
                checkpoints = {
                    'QA_model': self.QA_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_acc': best_val_acc,
                    'last_epoch': epoch + 1,
                }
                torch.save(checkpoints, os.path.join(self.QA_logs_path, 'checkpoint.pt'))

    @torch.no_grad()
    def eval(self, dl_val: DataLoader, ref_arr=None):
        scaler = torch.cuda.amp.GradScaler()  # Scaler for mixed precision (16/32 bit) computation
        # Set model to evaulation mode
        self.QA_model.eval()
        val_loss = 0  # Initialize validation loss
        val_acc = 0  # Initialize validation accuracy
        val_top5_acc = 0  # Initialize validation top5 accuracy
        val_bleu = 0  # Initialize validation bleu_score
        with torch.no_grad():
            # Get query and key image batches
            for val_data in dl_val:
                Q = val_data['Q']  # Question
                A = val_data['A']  # Answer
                A_Labels = val_data['A_Labels']  # Answer labels
                A_Labels = torch.LongTensor(A_Labels.type(torch.int64)).to(self.device)  # Cast to long
                Im = val_data['Im_Tensor'].to(self.device)  # Send image to device

                # Get logits, labels and then loss
                with torch.cuda.amp.autocast():
                    outputs = self.QA_model(Q, A, Im, A_Labels)
                    loss = outputs[0]

                # Get logits from model
                logits = outputs[1]
                # Get predictions
                pred = torch.argmax(logits, dim=1)
                # Get accuracy
                val_acc += torch.mean((pred == A_Labels).type(torch.float))
                val_loss += float(loss)

                # For top5 accuracy metric
                y_score = logits.cpu().detach().numpy()
                y_true = A_Labels.cpu().detach().numpy()
                val_top5_acc += top_k_accuracy_score(y_true, y_score, k=5, labels=range(517))

                # Reset loss to None to save memory
                loss = None

                # Calculate BLEU score. N in N-grams changes based on the length of the answer. For example, 4-gram for
                # the answe 'no' is nonsensical. Weights are equally distributed (i.e. for 4-gram, weight is [0.25] * 4)
                temp = 0
                for i in range(len(y_true)):
                    reference = [ref_arr[y_true[i]].split(' ')]
                    predict = ref_arr[pred[i]].split(' ')
                    n = min(4, len(reference[0]), len(predict))
                    weights = tuple([1. / n] * n)
                    temp += sentence_bleu(reference, predict, weights=weights)
                val_bleu += temp / len(val_data)
                # Return metrics
            return val_loss / len(dl_val), val_acc / len(dl_val), val_top5_acc / len(dl_val), val_bleu / len(dl_val)
