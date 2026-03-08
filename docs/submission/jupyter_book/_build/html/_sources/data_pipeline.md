# Data Pipeline

## Dataset
PlantVillage in folder-per-class structure under `ml/datasets/plantvillage`.

## Split Strategy
`ml/src/data/split_data.py` creates stratified train/val/test splits (80/10/10).

## Preprocessing and Augmentation
Defined in `ml/src/data/augmentations.py`:
- resize
- horizontal flip
- rotation
- zoom/scale
- brightness adjustment

## Class Imbalance
`ml/src/data/class_weights.py` computes class weights used in weighted cross-entropy.
