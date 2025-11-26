```
import argparse
import torch
from torch.utils.data import Dataset DataLoader
from src.utils import read_config ensure_dir load_checkpoint
from src.models import DescriptorEncoder FewShotClassifier
import torch.nn.functional as F


class FewShotDataset(Dataset):
def __init__(self, data_list):
self.data = data_list
def __len__(self):
return len(self.data)
def __getitem__(self, idx):
x,y = self.data[idx]
return x,y




def finetune(cfg):
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained = load_checkpoint(cfg['pretrained_checkpoint'], map_location=device)
encoder = DescriptorEncoder(input_dim=cfg['embedding_dim'], emb_dim=cfg['embedding_dim'])
encoder.load_state_dict(pretrained['encoder'])
encoder.to(device)
encoder.eval()


classifier = FewShotClassifier(emb_dim=cfg['embedding_dim'], num_classes=cfg.get('num_classes', 10)).to(device)
optimizer = torch.optim.AdamW(classifier.parameters(), lr=cfg['learning_rate'])


# prepare few shot data loader - placeholder
train_data = []
val_data = []
train_loader = DataLoader(FewShotDataset(train_data), batch_size=cfg['batch_size'], shuffle=True)


for epoch in range(cfg['num_epochs']):
classifier.train()
for x,y in train_loader:
x = x.to(device)
y = y.to(device)
with torch.no_grad():
feat = encoder(x)
logits = classifier(feat)
loss = F.cross_entropy(logits, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
# save final classifier
torch.save({'classifier': classifier.state_dict()}, cfg['output_root'] + '/finetuned_classifier.pt')


if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()
cfg = read_config(args.config)
finetune(cfg)
```
