# How to Clone UniRel

Run commands one at a time in this directory.

## Anaconda Setup

```sh
# Anaconda setup
conda create --name UniRel
conda activate UniRel
```

### CUDA

If working with CUDA, make sure CUDA toolkit is installed:

```sh
conda install --channel nvidia cudatoolkit
```

### PyTorch

Use [PyTorch's tool](https://pytorch.org/get-started/locally/) to generate the appropriate command for installing PyTorch.

For example:

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Other Dependencies

```sh
conda install --channel conda-forge accelerate gdown ipykernel ipywidgets numpy scikit-learn
conda install --channel huggingface transformers
```

## Accelerate Setup

See [Configuring 🤗 Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/install#configuring-accelerate) for details. If you don't know what to do, simply run the following Python code:

```py
from accelerate.utils import write_basic_config
write_basic_config(mixed_precision='fp16')
```

## Jupyter Setup

```sh
python -m ipykernel install --user --name=UniRel
```

## Download Data

```sh
gdown 1-3uBc_VfaCEWO2_FegzSyBXNeFmqhv7x --output data.zip
unzip data.zip "data4bert/*"
rm data.zip
mv data4bert data
```

## Download Checkpoint

```sh
gdown --folder 1poRbtpm5ddbwUk3mVQ2-4G_o3OPXjYNq --output ./output/nyt/checkpoint-final
```
