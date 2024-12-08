import os
from time import time
import torch
import argparse
import h5py
import numpy as np
import scanpy as sc
from torch.utils.data import TensorDataset, random_split
from sklearn import preprocessing

from model import spaVAE
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dtype', default=torch.float64)
parser.add_argument('--data_file', default='/home/math/pjiangag/spaVAE/datasets/sample_151673.h5')
parser.add_argument('--train_ratio', default=0.95, type=float)
parser.add_argument('--patience', default=200, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--weight_decay', default=1e-6, type=float)
parser.add_argument('--dropoutE', default=0, type=float,
                    help='dropout probability for encoder')
parser.add_argument('--dropoutD', default=0, type=float,
                    help='dropout probability for decoder')
parser.add_argument('--encoder_layers', nargs="+", default=[128, 64], type=int)
parser.add_argument('--GP_dim', default=2, type=int,help='dimension of the latent Gaussian process embedding')
parser.add_argument('--Normal_dim', default=8, type=int,help='dimension of the latent standard Gaussian embedding')
parser.add_argument('--decoder_layers', nargs="+", default=[128], type=int)
parser.add_argument('--init_beta', default=10, type=float, help='initial coefficient of the KL loss')
parser.add_argument('--min_beta', default=4, type=float, help='minimal coefficient of the KL loss')
parser.add_argument('--max_beta', default=25, type=float, help='maximal coefficient of the KL loss')
parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--inducing_point_steps', default=10, type=int)
parser.add_argument('--inducing_point_nums', default=None, type=int)
parser.add_argument('--fixed_gp_params', default=False, type=bool)
parser.add_argument('--loc_range', default=20., type=float)
parser.add_argument('--kernel_scale', default=20., type=float)
parser.add_argument('--model_file', default='model.pt')
parser.add_argument('--final_latent_file', default='final_latent.txt')
parser.add_argument('--denoised_counts_file', default='denoised_counts.txt')
parser.add_argument('--num_denoise_samples', default=10000, type=int)
parser.add_argument('--device', default='cuda')

args = parser.parse_args()

if __name__ == "__main__":
    print(args)

    eps = 1e-5
    initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
    print(initial_inducing_points.shape)

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

    dataset = TensorDataset(torch.tensor(X, dtype=args.dtype), torch.tensor(ann_data.X, dtype=args.dtype), 
                            torch.tensor(ann_data.raw.X, dtype=args.dtype), torch.tensor(ann_data.obs.size_factors, dtype=args.dtype))

    train_dataset, val_dataset = random_split(dataset=dataset, lengths=[args.train_ratio, 1.-args.train_ratio])
    print(f"train size: {train_dataset.__len__()}")
    print(f"test size:{val_dataset.__len__()}")

    model = spaVAE(input_dim=ann_data.n_vars, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=ann_data.n_obs, KL_loss=args.KL_loss, dynamicVAE=args.dynamicVAE, 
        init_beta=args.init_beta, min_beta=args.min_beta, max_beta=args.max_beta, dtype=torch.float64, device=args.device)
    
    print(str(model))

    if not os.path.isfile(args.model_file):
        t0 = time.time()
        model.train(pos=X, ncounts=ann_data.X, raw_counts=ann_data.raw.X, size_factors=ann_data.obs.size_factors,
                lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                train_size=args.train_ratio, maxiter=args.maxiter, patience=args.patience, save_model=True, model_weights=args.model_file)
        print('Training time: %d seconds.' % int(time() - t0))
    else:
        model.load_model(args.model_file)

    final_latent = model.sample_latent_embedding(X=X, Y=ann_data.X, batch_size=args.batch_size)
    np.savetxt(args.final_latent_file, final_latent, delimiter=",")
