```
#!/usr/bin/env bash
# create conda env and install dependencies
ENV_NAME=gs_contrast
PYTHON=python3
conda create -y -n ${ENV_NAME} ${PYTHON}
conda activate ${ENV_NAME}
pip install -r requirements.txt
echo "Please follow instructions in README to install the official 3D Gaussian Splatting repository as a dependency"
```
