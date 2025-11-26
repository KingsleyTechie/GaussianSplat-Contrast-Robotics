

import torch.nn as nn


class DescriptorEncoder(nn.Module):
def __init__(self, input_dim=256, emb_dim=256):
super().__init__()
self.net = nn.Sequential(
nn.Linear(input_dim, 512),
nn.ReLU(),
nn.Linear(512, emb_dim)
)


def forward(self, x):
return self.net(x)


class ProjectionHead(nn.Module):
def __init__(self, emb_dim=256, proj_dim=128):
super().__init__()
self.net = nn.Sequential(
nn.Linear(emb_dim, emb_dim),
nn.ReLU(),
nn.Linear(emb_dim, proj_dim)
)


def forward(self, x):
return self.net(x)


class FewShotClassifier(nn.Module):
def __init__(self, emb_dim=256, num_classes=10):
super().__init__()
self.net = nn.Sequential(
nn.Linear(emb_dim, 128),
nn.ReLU(),
nn.Linear(128, num_classes)
)


def forward(self, x):
return self.net(x)
```
