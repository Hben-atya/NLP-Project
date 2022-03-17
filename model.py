import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.models import efficientnet_b0, efficientnet_b6
from transformers import AutoModelForSequenceClassification, \
    AutoTokenizer
from dataset import logits_to_str


class ClsModel(pl.LightningModule):
    # Model for image classification
    def __init__(self):
        super().__init__()

        # EfficientNet B0 pretrained on ImageNet is sufficient
        self.model = efficientnet_b0(pretrained=True)

        # Initialize a FC layer to get the required number of classes
        self.fc = nn.Linear(in_features=1000, out_features=3, bias=True)

        # Initialize softmax layer
        self.softmax = nn.Softmax(dim=1)

        # Initialize loss (CELoss)
        self.crossentropy = nn.CrossEntropyLoss()

    def forward(self, x):
        # Forward
        x = self.model(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def shared_step(self, batch, stage):
        # Shared step (pytorch-lightning format)
        logits = self.forward(batch[0])
        y = batch[1]

        loss = self.crossentropy(logits, y)

        acc = (torch.argmax(logits, dim=1) == y).float().mean()
        return {
            "loss": loss,
            "acc": acc
        }

    def shared_epoch_end(self, outputs, stage):
        # Shared epoch end (pytorch-lightning format)
        acc = outputs[1]
        metrics = {
            f"{stage}_acc": acc
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00001)


class QAModel(nn.Module):
    def __init__(self, model_args, device=None):
        super(QAModel, self).__init__()
        self.device = device  # Set device

        self.Q_Classifier_Path = model_args['Q_Classifier_Path']  # Path to trained Question Classification model
        self.A_Classifier_Path = model_args['A_Classifier_Path']  # Path to trained Answer Classification model
        self.Im_Classifier_Path = model_args['Im_Classifier_Path']  # Path to trained Image Classification model

        self.N = model_args['batch_size']  # Batch size

        # Load sequence classification model pretrained on BioMed data
        self.model = AutoModelForSequenceClassification.from_pretrained("mmoradi/Robust-Biomed-RoBERTa-TextClassification",
                                                                        num_labels=517)
        self.model.to(self.device)  # Send model to device

        self.embedder = self.model.roberta.embeddings.word_embeddings  # Extract embedder
        self.embedder.to(self.device)  # Send embedder to device

        # Load tokenizer pretrained on BioMed data
        self.tokenizer = AutoTokenizer.from_pretrained("mmoradi/Robust-Biomed-RoBERTa-TextClassification")

        # Set Loss as CELoss
        self.CrossEntropy = nn.CrossEntropyLoss()

        # Initialize Image encoder as EfficientNet B6 pretrained on ImageNet
        self.image_encoder = efficientnet_b6(pretrained=True).features
        self.image_encoder.to(device=device)  # Send encoder to device

        # For feature map extraction, initialize conv layers that take each map and output them all with 768 channels
        self.conv1 = nn.Conv2d(in_channels=144, out_channels=768, kernel_size=1).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=200, out_channels=768, kernel_size=1).to(self.device)
        self.conv3 = nn.Conv2d(in_channels=344, out_channels=768, kernel_size=1).to(self.device)
        self.conv4 = nn.Conv2d(in_channels=576, out_channels=768, kernel_size=1).to(self.device)
        self.conv5 = nn.Conv2d(in_channels=2304, out_channels=768, kernel_size=1).to(self.device)

        # Initialize adaptive avg 2D pooling layer. This takes the result of the conv layers and make them 1x1x768x1
        self.avg = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)

        # In order to input the concatenated embeddings in a way that the model understands, we need to find the
        # cls token embedding which begins the sentence, and the sep token that separates between sentences or sequences
        self.cls_token_emb = self.embedder(torch.tensor([0]).to(self.device))
        self.sep_token_emb = self.embedder(torch.tensor([2]).to(self.device))

        # Load question and answer classifiers from the respective paths
        self.question_classifier = AutoModelForSequenceClassification.from_pretrained(self.Q_Classifier_Path)
        self.answer_classifier = AutoModelForSequenceClassification.from_pretrained(self.A_Classifier_Path)

        # Initialize image classification model
        self.image_classifier = ClsModel()

        # Load image classification model from path
        im_cls_checkpoint = torch.load(self.Im_Classifier_Path)
        self.image_classifier.load_state_dict(im_cls_checkpoint['state_dict'])
        self.image_classifier.to(self.device)  # Send to device

    def forward(self, Q, A, Im, A_Labels):
        with torch.no_grad():
            # Get the feature maps from image encoder
            feat1 = self.image_encoder[:5](Im)
            feat2 = self.image_encoder[5](feat1)
            feat3 = self.image_encoder[6](feat2)
            feat4 = self.image_encoder[7](feat3)
            feat5 = self.image_encoder[8](feat4)

        # Get feature embeddings after conv and avg-pooling layers and concatenate them
        feat_embeds = self.avg(self.conv1(feat1).squeeze(-1))
        feat_embeds = torch.concat((feat_embeds, self.avg(self.conv2(feat2).squeeze(-1))), dim=2)
        feat_embeds = torch.concat((feat_embeds, self.avg(self.conv3(feat3).squeeze(-1))), dim=2)
        feat_embeds = torch.concat((feat_embeds, self.avg(self.conv4(feat4).squeeze(-1))), dim=2)
        feat_embeds = torch.concat((feat_embeds, self.avg(self.conv5(feat5).squeeze(-1))), dim=2).view(self.N, 5, 768)

        # Tokenize Question
        tokenized_Q = self.tokenizer(Q,
                                     return_tensors='pt',
                                     padding=True,
                                     truncation=True
                                     )

        # We want all of the pretrained models (other than the QA model) to not train during the VQA training
        with torch.no_grad():

            # Get Question classification ID
            Q_cls_ids = self.tokenizer(logits_to_str(self.question_classifier(tokenized_Q['input_ids'].to(self.device)).logits, A_Q_Im='Q'),
                                       return_tensors='pt',
                                       padding=True,
                                       truncation=True)['input_ids'].to(self.device)

            # Make the first token the sep token
            Q_cls_ids[0] = 2

            # Get Answer classification ID
            A_cls_ids = self.tokenizer(logits_to_str(self.answer_classifier(tokenized_Q['input_ids'].to(self.device)).logits, A_Q_Im='A'),
                                       return_tensors='pt',
                                       padding=True,
                                       truncation=True)['input_ids'].to(self.device)
            # Remove cls ID
            A_cls_ids = A_cls_ids[:, 1:]

            # Get Image classification IDs
            Im_cls_ids = self.tokenizer(logits_to_str(self.image_classifier(Im), A_Q_Im='Im'),
                                        return_tensors='pt',
                                        padding=True,
                                        truncation=True)['input_ids'].to(self.device)

            # Remove cls ID
            Im_cls_ids = Im_cls_ids[:, 1:]

        # Get the embeddings from the various classification IDs
        Q_cls_embed = self.embedder(Q_cls_ids)
        A_cls_embed = self.embedder(A_cls_ids)
        Im_cls_embed = self.embedder(Im_cls_ids)

        # Get the IDs and the embeddings from the question
        Q_input_ids = tokenized_Q['input_ids'].to(self.device)
        input_embeds = self.embedder(Q_input_ids)

        # Concatenate all of the information that we've accumulated up until now
        input_embeds = torch.concat((
            input_embeds,  # Question embeddings
            feat_embeds,  # Feature map embeddings
            Q_cls_embed,  # Question class embeddings
            A_cls_embed,  # Answer class embeddings
            Im_cls_embed  # Image class embeddings
        ), dim=1)

        # The input to the model
        inputs = {
            'inputs_embeds': input_embeds.to(self.device),
            'labels': A_Labels.to(self.device)
        }

        # The output of the model
        outputs = self.model(**inputs)
        return outputs
