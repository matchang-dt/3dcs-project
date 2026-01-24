# Dataset Configurations

This directory contains Hydra configuration files for different datasets.

## Available Datasets

### ACID Dataset
- **acid_train.yaml**: Training split
- **acid_val.yaml**: Validation split
- **acid_test.yaml**: Test split

### RE10K Dataset
- **re10k_train.yaml**: Training split
- **re10k_test.yaml**: Test split

## Configuration Structure

All dataset configs follow this structure:

```yaml
_target_: datasets.dataset.DatasetCfg

name: acid  # or 're10k'
data_root: /workspace/re10kvol/acid
stage: train  # or 'validation', 'test'

# View sampling configuration
num_input_views: 2
num_target_views: 4

# Image configuration
target_image_size: 256

# Training schedule
max_train_steps: 300000  # For baseline expansion (0 for eval)
```

## Usage

Include a dataset config in your training config:

```yaml
# In configs/train/my_experiment.yaml
defaults:
  - /dataset/acid_train

# Override specific parameters if needed
dataset:
  num_input_views: 4
  target_image_size: 512
```

Or use from command line:

```bash
python train.py dataset=acid_train
python train.py dataset=re10k_train dataset.target_image_size=512
```

## Parameters

- **name**: Dataset identifier ('acid' or 're10k')
- **data_root**: Path to dataset root directory containing {train,test,validation}/ subdirectories
- **stage**: Data split ('train', 'test', or 'validation')
- **num_input_views**: Number of context/input views (typically 2)
- **num_target_views**: Number of target views for supervision (typically 4)
- **target_image_size**: Resize images to this size (height=width, default: 256)
- **max_train_steps**: Maximum training steps for baseline expansion schedule (set to 0 for evaluation)

## Data Format

Both ACID and RE10K datasets share the same format:
- Directory structure: `{data_root}/{stage}/*.torch`
- Each .torch file contains a list of 10 scenes
- Each scene contains: url, timestamps, cameras, images (JPEG-compressed), key

