

```
# package initializer
```


src/utils.py


```
import os
import yaml
import torch
from pathlib import Path




def read_config(path):
with open(path) as f:
return yaml.safe_load(f)




def ensure_dir(path):
Path(path).mkdir(parents=True, exist_ok=True)




def save_checkpoint(model_state, path):
torch.save(model_state, path)




def load_checkpoint(path, map_location=None):
return torch.load(path, map_location=map_location)
```
