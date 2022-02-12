import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from tqdm import tqdm


class MoCo_Trainer:
    def __init__(self, moco_model, moco_model_args, optimizer=None, device=None):
        self.moco_model = moco_model  # MoCo model
        self.f_q = moco_model.f_q  # Query encoder
        self.f_k = moco_model.f_k  # Key encoder
        self.optimizer = optimizer  # Optimizer
        self.device = device  # Device
        self.moco_model_args = moco_model_args  # MoCo model arguments
        self.moco_logs_path = moco_model_args['log_path']  # MoCo model logging path
        self.writer = SummaryWriter(self.moco_logs_path)  # Summary writer (tensorboard) for logging

        # Sending encoders to device
        if self.device:
            self.f_q.to(self.device)
            self.f_k.to(self.device)

    def fit(self,
            dl_train: DataLoader,
            dl_dev: DataLoader
            ):
        scaler = torch.cuda.amp.GradScaler()  # Scaler for mixed precision (16/32 bit) computation
        # Set both encoders to training mode
        self.f_q.train()
        self.f_k.train()
        num_epochs = self.moco_model_args['num_epochs']  # Number of epochs
        val_step = self.moco_model_args['val_step']  # Validate every val_step steps
        save_every = self.moco_model_args['save_every']  # Save every save_every steps
        best_val_loss = 1e8  # Initialize best validation loss
        last_epoch = 0  # Initialize the last epoch (for train resuming purposes)

        # If resume_run argument is True, load last checkpoint and continue training from there
        if self.moco_model_args['resume_run'] and os.path.exists(os.path.join(self.moco_logs_path, 'checkpoint.pt')):
            checkpoints = torch.load(os.path.join(self.moco_logs_path, 'checkpoint.pt'))
            self.f_k = checkpoints['fk_model']
            self.f_q = checkpoints['fq_model']
            self.optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
            best_val_loss = checkpoints['best_val_loss']
            last_epoch = checkpoints['last_epoch']
        gamma = 0.5  # Gamma value for learning rate scheduler
        # Initialize learning rate scheduler (simple step scheduler)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=50,
                                                    gamma=gamma)
        # Initialize criterion (cross entropy loss)
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(last_epoch, num_epochs):
            # Set model to training mode
            self.moco_model.train()
            startTime = time.time()  # Start time
            train_epoch_loss = 0  # Initialize epoch loss
            # Get query and key image batches
            for batch_data in dl_train:
                images_q = batch_data['image1'].to(self.device)
                images_k = batch_data['image2'].to(self.device)

                # Get logits, labels and then loss
                with torch.cuda.amp.autocast():
                    logits, labels = self.moco_model(images_q, images_k)
                    loss = criterion(logits, labels)

                # Updates
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                train_epoch_loss += float(loss / 128)
                loss = None

            # Update learning rate scheduler
            scheduler.step()
            endTime = time.time()  # End time
            torch.cuda.empty_cache()  # Empty cache
            avg_train_loss = train_epoch_loss / len(dl_train)  # Get average training loss

            # Print statistics
            print('epoch {}, loss {}, epoch time {} seconds, time remaining {} hours'.format(
                epoch + 1,
                avg_train_loss,
                round(endTime - startTime, 2),
                round(num_epochs * (endTime - startTime) / 3600 - epoch * (endTime - startTime) / 3600, 2)
            ))

            # Log, Save, and Evaluate
            self.writer.add_scalar(tag='Loss/train_loss', scalar_value=avg_train_loss, global_step=epoch + 1)

            if (epoch + 1) % val_step == 0:
                avg_val_loss = self.eval(dl_dev)
                self.writer.add_scalar(tag='Loss/val_loss', scalar_value=avg_val_loss, global_step=epoch + 1)
                print('epoch {}, val loss {}'.format(epoch + 1, avg_val_loss))
                if avg_val_loss < best_val_loss:
                    print('save new best model')
                    torch.save(self.f_q.state_dict(), os.path.join(self.moco_logs_path, 'best_fq_model.pt'))
                    torch.save(self.f_k.state_dict(), os.path.join(self.moco_logs_path, 'best_fk_model.pt'))
                    best_val_loss = avg_val_loss

            if (epoch + 1) % save_every == 0:
                checkpoints = {
                    'fq_model': self.f_q.state_dict(),
                    'fk_model': self.f_k.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'last_epoch': epoch + 1,
                }
                torch.save(checkpoints, os.path.join(self.moco_logs_path, 'checkpoint.pt'))

    @torch.no_grad()
    def eval(self, dl_val: DataLoader):
        scaler = torch.cuda.amp.GradScaler()  # Scaler for mixed precision (16/32 bit) computation
        # Set model to evaulation mode
        self.moco_model.eval()
        val_loss = 0  # Initialize validation loss
        # Initialize criterion (cross entropy loss)
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            # Get query and key image batches
            for val_data in dl_val:
                images_q = val_data['image1'].to(self.device)
                images_k = val_data['image2'].to(self.device)

                # Get logits, labels and then loss
                with torch.cuda.amp.autocast():
                    logits, labels = self.moco_model(images_q, images_k)
                    loss = criterion(logits, labels).detach()
                val_loss += float(loss) / 128
                loss = None

            return val_loss / len(dl_val)
