import os
import time
import torch
import argparse
import h5py
import numpy as np
import scanpy as sc
from torch.utils.data import TensorDataset, random_split
from sklearn import preprocessing

from model import spaVAE
from utils import *

eps = 1e-5
torch.manual_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--dtype', default=torch.float64)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--latent_GP_dim', default=2, type=int)
parser.add_argument('--latent_Gau_dim', default=8, type=int)
parser.add_argument('--encoder_layers', nargs="+", default=[128, 64], type=int)
parser.add_argument('--decoder_layers', nargs="+", default=[128], type=int)
parser.add_argument('--dropoutE', default=0, type=float)
parser.add_argument('--dropoutD', default=0, type=float)
parser.add_argument('--kernel_scale', default=20., type=float)
parser.add_argument('--desired_KL_loss', default=0.025, type=float)
parser.add_argument('--init_beta', default=10, type=float)
parser.add_argument('--min_beta', default=4, type=float)
parser.add_argument('--max_beta', default=25, type=float)

parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--weight_decay', default=1e-6, type=float)
parser.add_argument('--train_ratio', default=0.95, type=float)
parser.add_argument('--maxiter', default=5000, type=int)
parser.add_argument('--patience', default=200, type=int)

parser.add_argument('--inducing_point_steps', default=10, type=int)
parser.add_argument('--loc_range', default=20., type=float)

parser.add_argument('--data_file', default='/home/pjiangag/main/my_spaVAE/datasets/sample_151673.h5')

args = parser.parse_args()

model_name = args.data_file.split('/')[-1].split('.')[0]
checkpoints_path = f'/home/pjiangag/main/my_spaVAE/checkpoints/{model_name}.pt'

if __name__ == "__main__":

    print(args)

    initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
    print(initial_inducing_points.shape)

    # load data
    data = h5py.File(args.data_file, 'r')
    Y = np.array(data['X']).astype('float64')
    X = np.array(data['pos']).astype('float64')
    data.close()

    print(f"Y: {Y.shape}")
    print(f"X: {X.shape}")

    # rescale coordinates
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X) * args.loc_range # This value can be set larger if it isn't numerical stable during training.

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
    
    model = spaVAE(
        dtype=args.dtype,
        device=args.device,
        checkpoints_path=checkpoints_path,
        input_dim=ann_data.n_vars,
        latent_GP_dim=args.latent_GP_dim,
        latent_Gau_dim=args.latent_Gau_dim,
        encoder_dims=args.encoder_layers,
        decoder_dims=args.decoder_layers,
        encoder_dropout=args.dropoutE,
        decoder_dropout=args.dropoutD,
        initial_inducing_points=initial_inducing_points,
        initial_kernel_scale=args.kernel_scale,
        desired_KL_loss=args.desired_KL_loss,
        init_beta=args.init_beta,
        min_beta=args.min_beta,
        max_beta=args.max_beta,
        N_train=ann_data.n_obs
    )
    
    print(str(model))

    t0 = time.time()
    model.train_model(pos=X, ncounts=ann_data.X, raw_counts=ann_data.raw.X, size_factors=ann_data.obs.size_factors,
                    lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                    train_ratio=args.train_ratio, maxiter=args.maxiter, patience=args.patience)
    print('Training time: %d seconds.' % int(time.time() - t0))
