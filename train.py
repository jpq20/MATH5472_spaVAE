import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model import spaVAE
from utils import *


torch.manual_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', default='sample_151672')

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
parser.add_argument('--weight_decay', default=1e-6, type=float)
parser.add_argument('--train_ratio', default=0.95, type=float)
parser.add_argument('--maxiter', default=5000, type=int)
parser.add_argument('--patience', default=200, type=int)

parser.add_argument('--inducing_point_steps', default=10, type=int)
parser.add_argument('--loc_range', default=20., type=float)

args = parser.parse_args()

data_file = f'/home/pjiangag/main/my_spaVAE/datasets/{args.model_name}.h5'
checkpoints_path = f'/home/pjiangag/main/my_spaVAE/checkpoints/{args.model_name}.pt'
latent_file = f'/home/pjiangag/main/my_spaVAE/checkpoints/{args.model_name}_latent.txt'


if __name__ == "__main__":

    ann_data, X, initial_inducing_points = load_data(data_file, args.loc_range, args.inducing_point_steps)
    
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
    print(args)

    elbo_list = model.train_model(pos=X, ncounts=ann_data.X, raw_counts=ann_data.raw.X, size_factors=ann_data.obs.size_factors,
                    lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
                    train_ratio=args.train_ratio, maxiter=args.maxiter, patience=args.patience)

    final_latent = model.sample_latent_embedding(X=X, Y=ann_data.X, batch_size=args.batch_size)
    np.savetxt(latent_file, final_latent, delimiter=",")

    # plot the ELBO
    plt.plot(elbo_list)
    plt.savefig(f'/home/pjiangag/main/my_spaVAE/checkpoints/{args.model_name}_elbo.png')
