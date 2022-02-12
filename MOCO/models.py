import torch
from torch import nn
from torchvision.models import efficientnet_b0
import copy


class Encoder(nn.Module):
    def __init__(self, feat_dim, original_model):
        super(Encoder, self).__init__()
        self.original_model = original_model  # Original encoder model
        # It was decided to use EfficientNet B0, as it has the expressive power of ResNet50,
        # but with way less parameters and a faster training time
        num_ftrs = 1280  # Feature dimension in output (before FC layer) of EfficientNet B0

        # Replace FC layer of EfficientNet B0 with the following linear layers
        self.original_model.classifier = nn.Sequential(nn.Linear(num_ftrs, num_ftrs),
                                                       nn.ReLU(),
                                                       nn.Linear(num_ftrs, feat_dim)
                                                       )

    def forward(self, x):
        # Forward
        x = self.original_model(x)
        return x


class MoCoV2(nn.Module):
    def __init__(self, moco_args):
        super(MoCoV2, self).__init__()

        self.queue_len = moco_args['K']  # Queue length
        self.feat_dim = moco_args['feat_dim']  # Feature dimension for encoder output
        self.moco_args = moco_args  # MoCo model arguments
        original_model = efficientnet_b0(pretrained=False)  # Original encoder: Pretrained EfficientNet B1
        if moco_args['to_grayscale']:
            original_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # Query encoder initialization
        self.f_q = Encoder(feat_dim=self.feat_dim, original_model=original_model)
        # Key encoder (copy of query encoder)
        self.f_k = copy.deepcopy(self.f_q)

        # Copying all parameters from query encoder to key encoder and setting no gradients on key encoder
        for theta_q, theta_k in zip(self.f_q.parameters(), self.f_k.parameters()):
            theta_k.data.copy_(theta_q.data).detach()
            theta_k.requires_grad = False

        # Initialize and normalize queue
        self.register_buffer("queue", torch.randn(self.feat_dim, self.queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0).cuda().detach()

    @torch.no_grad()
    def momentum_update(self):
        m = self.moco_args['momentum']  # Momentum parameter for momentum update
        # Update key encoder parameters with momentum update
        for theta_q, theta_k in zip(self.f_q.parameters(), self.f_k.parameters()):
            theta_k.data = theta_k.data * m + theta_q.data * (1. - m)
            theta_k.requires_grad = False

    @torch.no_grad()
    def requeue(self, k):
        # Add latest batch to queue and remove earliest batch (enqueue and dequeue)
        self.queue = torch.t(torch.cat((torch.t(self.queue), k), dim=0))
        return self.queue[:, k.shape[0]:]

    def InfoNCELoss(self, q, k):
        T = self.moco_args['temperature']  # Temperature parameter
        N, C = q.shape  # Query shape
        K = self.queue_len  # Queue length

        # Positive samples
        l_positive = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).squeeze(-1)  # Nx1
        # Negative samples
        l_negative = torch.mm(q.view(N, C), self.queue.view(C, K))  # NxK

        # Make logits
        logits = torch.cat([l_positive, l_negative], dim=1) / T  # Nx(1+K)

        return logits

    def forward(self, im_q, im_k):
        # Get query and normalize it
        q = self.f_q(im_q)
        q = torch.nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            # Momentum update key encoder parameters
            self.momentum_update()
            # Get key and normalize it
            k = self.f_k(im_k)
            k = torch.nn.functional.normalize(k, dim=1).detach()

        # Get logits from query and key using InfoNCE loss function
        logits = self.InfoNCELoss(q, k)
        # Get labels
        labels = torch.zeros((q.shape[0],), dtype=torch.long).cuda()
        # Requeue the queue
        self.queue = self.requeue(k).detach()
        return logits, labels
