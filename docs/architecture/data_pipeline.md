# Data Pipeline

1. Input dataset in `ml/datasets/plantvillage/<class_name>/*.jpg`
2. Stratified split script creates `train.csv`, `val.csv`, `test.csv` in `ml/splits`
3. Class weights computed from `train.csv`
4. Three training scripts consume split CSV + class weights
5. Best weights saved under `ml/weights`
6. Evaluation scripts produce metrics/confusion matrix/benchmark artifacts
7. Backend inference loads selected model weights + class names
