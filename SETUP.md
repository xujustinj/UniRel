# How to Clone UniRel

## Initial Setup

Run commands one at a time in this directory.

```sh
# Anaconda setup
conda create --name UniRel --file requirements-conda.txt
conda activate UniRel

# Jupyter setup
python -m ipykernel install --user --name=UniRel

# Download checkpoint mentioned in README.md
gdown --folder 1poRbtpm5ddbwUk3mVQ2-4G_o3OPXjYNq --output ./output/nyt/checkpoint-final

# Download data mentioned in README.md
gdown 1-3uBc_VfaCEWO2_FegzSyBXNeFmqhv7x --output data.zip
unzip data.zip "data4bert/*"
rm data.zip
mv data4bert data
```

All scripts below assume that `UniRel` is active.

## After Adding Requirements

```sh
conda list --export > requirements-conda.txt
```

## Reinstalling Requirements

```sh
conda install --file requirements-conda.txt
```
