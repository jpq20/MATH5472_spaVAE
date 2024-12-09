import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import exp
import h5py
import scanpy as sc
from sklearn import preprocessing
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import umap


eps = 1e-10

class exp_trans(nn.Module):
    def __init__(self):
        super(exp_trans, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, modelfile='model.pt'):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.inf
        self.model_file = modelfile

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_checkpoints(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


class pid():
    """incremental PID controller"""
    def __init__(self, Kp, Ki, init_beta, min_beta, max_beta):
        """define them out of loop"""
        self.W_k1 = init_beta
        self.W_min = min_beta
        self.W_max = max_beta
        self.e_k1 = 0.0
        self.Kp = Kp
        self.Ki = Ki

    def _Kp_fun(self, Err, scale=1):
        return 1.0/(1.0 + float(scale)*exp(Err))

    def pid(self, exp_KL, kl_loss):
        """
        Incremental PID algorithm
        Input: KL_loss
        return: weight for KL divergence, beta
        """
        error_k = (exp_KL - kl_loss) * 5.   # we enlarge the error 5 times to allow faster tuning of beta
        ## comput U as the control factor
        dP = self.Kp * (self._Kp_fun(error_k) - self._Kp_fun(self.e_k1))
        dI = self.Ki * error_k

        if self.W_k1 < self.W_min:
            dI = 0
        dW = dP + dI
        ## update with previous W_k1
        Wk = dW + self.W_k1
        self.W_k1 = Wk
        self.e_k1 = error_k

        ## min and max value
        if Wk < self.W_min:
            Wk = self.W_min
        if Wk > self.W_max:
            Wk = self.W_max

        return Wk, error_k
    

class cauchy_kernel(nn.Module):
    """
    1/(1+dis^2/scale)
    """
    def __init__(self, scale, device, dtype=torch.float64,):
        super(cauchy_kernel, self).__init__()
        self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x1, x2):
        x1_square_sum = torch.sum(x1**2, dim=-1, keepdim=True)
        x2_square_sum = torch.sum(x2**2, dim=-1, keepdim=True)
        prod = torch.matmul(x1, torch.transpose(x2, -2, -1))
        dis2 = x1_square_sum + torch.transpose(x2_square_sum, -2, -1) - 2. * prod
        return 1/(1+dis2/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4)) # log(1+exp(sccale))
    
    def forward_1d(self, x1, x2):
        dis2 = torch.sum((x1-x2)**2, dim=1)
        return 1/(1+dis2/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
    

class nb_loss(nn.Module):
    def __init__(self):
        """
        compute the negative log-likelihood of gene NB distribution
        """
        super(nb_loss, self).__init__()

    def forward(self, x, mean, disp, scale_factor):
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        n_log_nb = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps) + \
            (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + x * (torch.log(disp+eps) - torch.log(mean+eps))
        return torch.sum(n_log_nb)
    

def gauss_cross_entropy(mu1, var1, mu2, var2, device):
    """
    multi-variate:
    H(p) = d/2*(1+log(2*pi)) + 0.5*log(|var1|)
    KL(p,q) = -d/2 + 0.5*(log(|var2|/|var1|) + tr(var2^{-1}@var1) + (mu2-mu1)^T@var2^{-1}@(mu2-mu1))
    1d-gaussian:
    H = 0.5 * log(2*pi) + 0.5 * log(var) + 0.5
    KL = log(var2/var1) + (var1 + (mu1-mu2)**2/var2)*0.5 - 0.5
    """
    return (-0.5 * (torch.log(2*torch.tensor([torch.pi], device=device)) + torch.log(var2) + (var1 + mu1 ** 2 - 2 * mu1 * mu2 + mu2 ** 2) / var2))
    
def build_network(layers, act="relu", dropout=0., norm="batchnorm"):
    net = []
    for i in range(1, len(layers)):
        # linear
        net.append(nn.Linear(layers[i-1], layers[i]))
        # norm
        if norm == "batchnorm":
            net.append(nn.BatchNorm1d(layers[i]))
        elif norm == "layernorm":
            net.append(nn.LayerNorm(layers[i]))
        # activation
        if act=="relu":
            net.append(nn.ReLU())
        elif act=="sigmoid":
            net.append(nn.Sigmoid())
        elif act=="elu":
            net.append(nn.ELU())
        # dropout
        if dropout > 0:
            net.append(nn.Dropout(p=dropout))
    return nn.Sequential(*net)

def load_data(data_file, loc_range, inducing_point_steps):
    data = h5py.File(data_file, 'r')
    Y = np.array(data['X']).astype('float64')
    X = np.array(data['pos']).astype('float64')
    data.close()

    # rescale coordinates
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X) * loc_range # This value can be set larger if it isn't numerical stable during training.

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

    inducing_points = np.mgrid[0:(1+eps):(1./inducing_point_steps), 0:(1+eps):(1./inducing_point_steps)].reshape(2, -1).T * loc_range
    
    return ann_data, X, inducing_points

def add_diagonal_jitter(matrix, jitter=1e-8):
    Eye = torch.eye(matrix.size(-1), device=matrix.device).expand(matrix.shape)
    return matrix + jitter * Eye

def refine(sample_id, pred, dis, shape="square"):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6 
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp.iloc[0:(num_nbs+1)]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred] >= num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
        if (i+1) % 1000 == 0:
            print("Processed", i+1, "lines")
    return np.array(refined_pred)