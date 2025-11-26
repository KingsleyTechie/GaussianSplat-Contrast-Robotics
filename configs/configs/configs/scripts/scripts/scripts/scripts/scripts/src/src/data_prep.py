```
import argparse
import os
from src.utils import read_config, ensure_dir


# This script prepares data from ScanNet or Matterport3D for the pipeline
# It expects the user to download datasets manually and place them under data_root.




def prepare_scannet(config):
# minimal instructions to prepare a ScanNet subset
data_root = config['data_root']
ensure_dir(data_root)
print('Prepare ScanNet data here')
# Real implementation: parse scannet scenes, extract rgb and depth frames and camera poses




def main():
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()
config = read_config(args.config)
prepare_scannet(config)


if __name__ == '__main__':
main()
```
