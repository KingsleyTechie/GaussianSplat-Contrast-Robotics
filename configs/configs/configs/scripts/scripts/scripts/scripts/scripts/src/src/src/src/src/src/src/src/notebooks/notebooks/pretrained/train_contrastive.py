import torch
import torch.nn as nn
import torch.optim as optim
from contrastive_encoder import ContrastiveEncoder

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long)

        return nn.CrossEntropyLoss()(logits, labels)

def train_step(encoder, optimizer, batch):
    anchor, positive, negative = batch
    anchor = anchor.float()
    positive = positive.float()
    negative = negative.float()

    z_anchor = encoder(anchor)
    z_pos = encoder(positive)
    z_neg = encoder(negative)

    loss_fn = InfoNCELoss()
    loss = loss_fn(z_anchor, z_pos, z_neg)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
