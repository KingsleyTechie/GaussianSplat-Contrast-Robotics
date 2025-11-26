```
Place pretrained weights here. Example filenames:
- contrastive_checkpoint.pt
- finetuned_classifier.pt


Due to file size pretrained weights are not included in this repository. You may upload them here after training locally or provide a download link in the paper.
```




Notes on integration with official 3D Gaussian Splatting repository


1. Clone the official 3DGS implementation into a subfolder outside the repository or install it as a dependency. I use the official repo for reconstruction and high fidelity fast rendering. The wrapper in src/splat_wrapper.py contains the minimal commands to call the reconstruction pipeline.


2. The 3DGS implementation may require CUDA specific builds and non standard python packages. Follow the original project instructions. After installation make sure the path to the 3DGS binaries is in your PATH or that you provide the correct python module path.


3. The descriptor extractor relies on the 3DGS renderer to produce small patches around splat projections. The wrapper assumes a command line interface exists to request rendering of patches per splat. If that is unavailable I provide a fallback renderer based on sampling rays through the splat and producing small synthetic patches.




Reproducibility and recommended hardware


- Single GPU with 24 GB VRAM is sufficient for small scale pretraining. For the full experiments similar to the paper I used two GPUs with 24 GB VRAM each to run larger batches and speed up pretraining.
- Reconstruction time varies by scene complexity 30 minutes to a few hours per scene on a single modern GPU.
- Pretraining time for the settings in configs/pretrain.yaml is on the order of several days to a week depending on batch size and number of scenes.




How I verified the results reported in the paper


1. I used ScanNet and Matterport3D subsets for training and evaluation. I followed dataset license and usage policies when downloading and using data. The scripts in data/README_DATA.md explain how to acquire and prepare the datasets.
2. I used the official 3DGS code to reconstruct scenes, then ran the descriptor extraction, contrastive training, and few shot evaluation using the scripts above.
3. For validation I compared results to baselines implemented in the repo. Baselines include a NeRF plus CNN pipeline and a point cloud contrastive baseline adapted from DepthContrast.




Support and contact


If you need help running the code or want assistance customizing training settings for your environment open an issue in the repository and I will help.
