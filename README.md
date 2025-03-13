# EyeReal

Realtime Glasses-Free 3D Display with Seamless Ultrawide Viewing Range using Deep Learning

## Installation

```bash
# Install torch
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Prepare CUDA
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# Install GS
cd lib/GS/submodules
git clone --recursive -b main https://github.com/ashawkey/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
git checkout d986da0d4cf2dfeb43b9a379b6e9fa0a7f3f7eea
git submodule update --init --recursive
cd ../../../..
pip install lib/GS/submodules/diff-gaussian-rasterization
pip install lib/GS/submodules/simple-knn
``` 

# Install other dependencies
```bash
pip install opencv-python tqdm plyfile wandb
```

## Preset before running scripts
```bash
# Windows
set PYTHONPATH=/path/to/EyeReal

# Linux
export PYTHONPATH=/path/to/EyeReal
```

## Generate light field
```bash
# Use lego_bulldozer as an example, other light fields like it

# First, you should have a .ply file corresponding to the desired light field
# We provide a weight file for lego_bulldozer at weight/gaussian_ply/lego_bulldozer.ply

# Next, you should select the viewpoints you want to observe and generate corresponding images
# Set parameters in data/data_preparation.py
# R represents physical world distance in centimeters
# phi indicates the angle between the axis perpendicular to your line of sight, theta denotes the angle between the coordinate axis aligned with the longer edge of the screen and the line of sight's projection on the ground
# Preset parameters in the file define viewpoint selection within a conical space: 
# - Front-facing position of lego_bulldozer
# - ±50° horizontal coverage (left/right)
# - ±30° vertical coverage (up/down)
# After configuring these settings, execute the program
python data/data_preparation.py
```

## Trainning
```bash
# After completing the previous steps, you should have the desired light field data images ready.
# you should modify the paramater --data_path and --scene in run.sh with your own
# Execute the modified script in run.sh the terminal
sh run.sh
```

## Inference
We provide two inference methods: one based on user-specified coordinates and another based on a pair of input images.
The model weights can be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1oQXisO1kS3MvihgCm090L-bAtywAspZF?usp=sharing) and put them in weight/model_ckpts.
### Inference Based on Coordinates
```bash
# Script: inference_coordinates.py
# - Update `eyeRealNet_weights` with your trained weights.
# - Modify `gaussian_path` to point to the .ply file corresponding to your light field data.
# - Set `mid_coord` to the desired viewpoint position in the world coordinate system (in centimeters).
python inference_coordinates.py
```
### Inference Based on Input Images
```bash
# Script: inference_figures.py
# - Update `eyeRealNet_weights` with your trained weights.
# - Provide a pair of images for the viewpoints by setting the paths for `data_path`, `left_eye_path`, and `right_eye_path`.
python inference_figures.py
```

## Eval
The data for evaluation can be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1guhUzAdIAYxNrWUJr0IiBfdaQqgzlejH?usp=sharing) and put them in dataset/eval.
```bash
# you can run eval\fullfield_benchmark.py to eval your model, we give an example in the file
# you can change ckpt_weights and args.data_path with your own 
python eval/fullfield_benchmark.py
```