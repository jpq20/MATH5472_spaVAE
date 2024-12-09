# my_spaVAE for MATH5472

## 1. files

- `model.py`: the model of spaVAE
- `train.py`: the training script
- `utils.py`: the utils for data loading and visualization
- `cluster.py`: the script for latent variables clustering
- `README.md`: the README file
- `datasets/`: the datasets for training and testing
- `checkpoints/`: the checkpoints and results for the model
- `requirements.txt`: the requirements for the project

## 2. How to run

### 2.1. Training

```bash
python train.py --model_name YOUR_MODEL_NAME
```

The model will be saved in the `checkpoints/` folder.

### 2.2. Clustering

```bash
python cluster.py --model_name YOUR_MODEL_NAME
```
The clustering results will be saved in the `checkpoints/` folder.

## 3. Clarification

Thanks a lot for the original code of spaVAE:https://github.com/ttgump/spaVAE/tree/main

I have made some modifications to the code to fit the requirements of MATH5472.
