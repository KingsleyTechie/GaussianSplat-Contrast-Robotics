```
self.files = descriptor_files
def __len__(self):
return len(self.files)
def __getitem__(self, idx):
# load descriptor pair or descriptor and augment
d = torch.load(self.files[idx])
return d




def contrastive_loss(z_a, z_b, temperature=0.1):
z_a = F.normalize(z_a, dim=-1)
z_b = F.normalize(z_b, dim=-1)
logits = torch.matmul(z_a, z_b.t()) / temperature
labels = torch.arange(z_a.size(0)).long().to(z_a.device)
loss = F.cross_entropy(logits, labels)
return loss




def train(cfg):
ensure_dir(cfg['output_root'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# placeholder for listing descriptor files
descriptor_files = []
dataset = DescriptorDataset(descriptor_files)
dataloader = DataLoader(dataset, batch_size=cfg['batch_size_descriptors'], shuffle=True)


encoder = DescriptorEncoder(input_dim=cfg['embedding_dim'], emb_dim=cfg['embedding_dim']).to(device)
projector = ProjectionHead(emb_dim=cfg['embedding_dim'], proj_dim=128).to(device)


optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(projector.parameters()), lr=cfg['lr'], weight_decay=cfg['weight_decay'])


for epoch in range(cfg['epochs']):
for batch in dataloader:
# batch should yield two augmented views per descriptor or pairs
# For illustration assume batch is tensor of size [B, 2, D]
batch = batch.to(device)
a = batch[:,0,:]
b = batch[:,1,:]
z_a = projector(encoder(a))
z_b = projector(encoder(b))
loss = contrastive_loss(z_a, z_b, temperature=cfg['temperature'])
optimizer.zero_grad()
loss.backward()
optimizer.step()
if (epoch + 1) % cfg.get('save_every', 5) == 0:
save_checkpoint({'encoder': encoder.state_dict(), 'projector': projector.state_dict()}, os.path.join(cfg['output_root'], f'checkpoint_epoch_{epoch+1}.pt'))




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()
cfg = read_config(args.config)
train(cfg)
```
