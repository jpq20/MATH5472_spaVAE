import numpy as np
import scipy as sp
import scanpy as sc
import seaborn as sns
import h5py
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch

# para
loc_scale = 20.
dtype = torch.float64
train_ratio = 0.95

# load data
data_path = "/Users/pqjiang/code/my_spaVAE/datasets/sample_151673.h5"
data = h5py.File(data_path, 'r')
Y = np.array(data['X']).astype('float64')
X = np.array(data['pos']).astype('float64')
data.close()

print(f"Y: {Y.shape}")
print(f"X: {X.shape}")

# rescale coordinates
scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X) * loc_scale # This value can be set larger if it isn't numerical stable during training.

# SCANPY to process
ann_data = sc.AnnData(Y, dtype="float64")
sc.pp.filter_genes(ann_data, min_counts=1)
sc.pp.filter_cells(ann_data, min_counts=1)

ann_data.raw = ann_data.copy()

sc.pp.normalize_per_cell(ann_data) # normalization

ann_data.obs['size_factors'] = ann_data.obs.n_counts / np.median(ann_data.obs.n_counts) # size factor in paper

# rescale
sc.pp.log1p(ann_data)
sc.pp.scale(ann_data)

dataset = TensorDataset(torch.tensor(X, dtype=dtype), torch.tensor(ann_data.X, dtype=dtype), 
                        torch.tensor(ann_data.raw.X, dtype=dtype), torch.tensor(ann_data.obs.size_factors, dtype=dtype))

train_dataset, val_dataset = random_split(dataset=dataset, lengths=[train_ratio, 1.-train_ratio])
print(f"train size: {train_dataset.__len__()}")
print(f"test size:{val_dataset.__len__()}")

torch.save(train_dataset, "./datasets/train_dataset.pt")
torch.save(val_dataset, "./datasets/val_dataset.pt")
