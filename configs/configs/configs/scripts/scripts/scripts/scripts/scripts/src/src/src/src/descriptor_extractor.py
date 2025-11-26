```
import argparse
import numpy as np
import torch
from src.utils import read_config, ensure_dir


# This module loads optimized 3DGS splat output and extracts per splat descriptors
# It renders small patches around each splat using the 3DGS renderer and computes geometry features




def extract_descriptors(scene_dir, out_dir, cfg):
# Basic pseudocode steps
# 1 Load splat parameters file produced by 3DGS
# 2 For each splat compute geometric features
# 3 For each splat render patches from m nearby views using the 3DGS renderer
# 4 Process patches through a small CNN to obtain appearance embedding
# 5 Save concatenated descriptors to disk
print('Extracting descriptors for', scene_dir)




def main():
parser = argparse.ArgumentParser()
parser.add_argument('--scene_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()
cfg = read_config(args.config)
ensure_dir(args.out_dir)
extract_descriptors(args.scene_dir, args.out_dir, cfg)


if __name__ == '__main__':
main()
```
