# GaussianSplat Contrast Robotics


This repository contains the full implementation to reproduce the experiments reported in my paper. I provide scripts and modules to run the entire pipeline from raw multi view captures to final few shot evaluation. I wrote the code so that you can run experiments on a single GPU or accelerate on two GPUs for larger pretraining batches.


Notes


1. I rely on the official 3D Gaussian Splatting repository for reconstruction and rendering. The codebase integrates with that project through a simple wrapper. The 3DGS implementation is heavy and requires a modern GPU. See the instructions below for installation.
2. Reproducing the full results requires time and GPU compute. I provide configuration files that match the experimental settings in the paper. I also include a smaller configuration for rapid prototyping.


Requirements


See requirements.txt for python dependencies. I recommend creating a conda environment using the provided setup script.


Quick start


1. Clone the repository and install dependencies.


```bash
git clone <your repo url>
cd GaussianSplat-Contrast-Robotics
bash setup_env.sh
```


2. Prepare datasets. See data README for steps to prepare ScanNet and Matterport3D subsets.


```bash
bash scripts/run_preprocess.sh --config configs/default.yaml
```


3. Run reconstruction for a scene using the 3DGS wrapper


```bash
bash scripts/run_reconstruct.sh --scene PATH_TO_SCENE --config configs/default.yaml
```


4. Run contrastive pretraining


```bash
bash scripts/run_pretrain.sh --config configs/pretrain.yaml
```


5. Run few shot fine tuning and evaluation


```bash
bash scripts/run_finetune.sh --config configs/finetune.yaml
bash scripts/run_eval.sh --config configs/finetune.yaml
```


Contact


If you find issues or need help reproducing results, open an issue in the repository and I will respond.
```
