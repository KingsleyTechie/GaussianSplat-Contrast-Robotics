```
import argparse
import torch
from src.utils import read_config


# Evaluation script computes few shot accuracy and detection mAP


def evaluate(cfg):
print('Evaluate with config', cfg)


if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()
cfg = read_config(args.config)
evaluate(cfg)
```
