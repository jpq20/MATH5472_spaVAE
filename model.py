import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from collections import deque
import math
from torch.utils.data import TensorDataset

from utils import *


class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu", dropout=0, dtype=torch.float32, norm="batchnorm"):
        super(encoder, self).__init__()
        self.layers = buildNetwork([input_dim]+hidden_dims, activation=activation, dropout=dropout, dtype=dtype, norm=norm)
        self.enc_mu = nn.Linear(hidden_dims[-1], output_dim)
        self.enc_var = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        h = self.layers(x)
        mu = self.enc_mu(h)
        var = torch.exp(self.enc_var(h).clamp(-15, 15))
        return mu, var


class decoder(nn.Module):
    def __init__(self, latent_dim, decoder_layers, activation='elu', dropout=0):
        super(decoder, self).__init__()
        self.layers = buildNetwork([latent_dim]+decoder_layers, activation=activation, dropout=dropout)

    def forward(self, x):
        return self.layers(x)


class spaVAE(nn.Module):
    def __init__(self, dtype, device, checkpoints_path,
                 input_dim, latent_GP_dim, latent_Gau_dim, encoder_dims, decoder_dims,
                 encoder_dropout, decoder_dropout,
                 initial_inducing_points, 
                 initial_kernel_scale, 
                 desired_KL_loss, init_beta, min_beta, max_beta,
                 N_train
                 ):
        super(spaVAE, self).__init__()

        torch.set_default_dtype(dtype)
        self.dtype = dtype
        self.device = device
        self.checkpoints_path = checkpoints_path

        self.input_dim = input_dim
        self.latent_GP_dim = latent_GP_dim
        self.latent_Gau_dim = latent_Gau_dim
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        self.kernel_scale = initial_kernel_scale

        self.desired_KL_loss = desired_KL_loss
        self.beta = init_beta
        self.PID = pid(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)

        self.encoder = encoder(input_dim=input_dim, hidden_dims=encoder_dims, output_dim=latent_GP_dim+latent_Gau_dim, activation="elu", dropout=encoder_dropout)
        self.decoder = decoder(latent_dim=latent_GP_dim+latent_Gau_dim, decoder_layers=decoder_dims, activation='elu', dropout=decoder_dropout)
        self.recon_mean = nn.Sequential(nn.Linear(decoder_dims[-1], input_dim), MeanAct())
        self.recon_covariance = nn.Parameter(torch.randn(self.input_dim), requires_grad=True)

        self.nbloss = nb_loss().to(self.device)

        self.N_train = N_train
        self.jitter = 1e-8

        self.inducing_index_points = torch.tensor(initial_inducing_points, dtype=dtype).to(device)
        self.kernel = cauchy_kernel(scale=initial_kernel_scale, device=device, dtype=dtype)

        self.to(self.device)

    def save_checkpoints(self):
        torch.save(self.state_dict(), self.checkpoints_path)

    def load_checkpoints(self, checkpoints_path):
        pretrained_dict = torch.load(checkpoints_path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def cal_L_H(self, x, y, noise, mu_hat,A_hat):
        """
        compute L_H for the data in the current batch
        """
        b = x.shape[0] # batch size
        p = self.inducing_index_points.shape[0] # number of inducing points

        K_pp = self.kernel(self.inducing_index_points, self.inducing_index_points) # (p,p)
        K_pp_inv = torch.linalg.inv(add_diagonal_jitter(K_pp, self.jitter)) # (p,p)

        K_nn = self.kernel.forward_1d(x, x) # (b,)

        K_np = self.kernel(x, self.inducing_index_points) # (b,p)
        K_pn = torch.transpose(K_np, 0, 1) # (p,b)

        # KL-term
        K_pp_chol = torch.linalg.cholesky(add_diagonal_jitter(K_pp, self.jitter)) # (p,p)
        S_chol = torch.linalg.cholesky(add_diagonal_jitter(A_hat, self.jitter)) # (p,p)
        K_pp_log_det = 2 * torch.sum(torch.log(torch.diagonal(K_pp_chol))) # log-det of K_pp
        S_log_det = 2 * torch.sum(torch.log(torch.diagonal(S_chol))) # log-det of A

        KL_term = 0.5 * (K_pp_log_det - S_log_det - p +
                             torch.trace(torch.matmul(K_pp_inv, A_hat)) +
                             torch.sum(mu_hat * torch.matmul(K_pp_inv, mu_hat)))

        # other terms in L_H
        mean_vector = torch.matmul(K_np, torch.matmul(K_pp_inv, mu_hat)) 

        precision = 1 / noise

        K_tilde_terms = precision * (K_nn - torch.diagonal(torch.matmul(K_np, torch.matmul(K_pp_inv, K_pn)))) # diag(K_nn - K_Np @ K_pp^{-1} @ K_pN)

        # Lambda
        lambda_mat = torch.matmul(K_np.unsqueeze(2), torch.transpose(K_np.unsqueeze(2), 1, 2))
        lambda_mat = torch.matmul(K_pp_inv, torch.matmul(lambda_mat, K_pp_inv))

        # Trace terms
        trace_terms = precision * torch.einsum('bii->b', torch.matmul(A_hat, lambda_mat))

        L_H = -0.5 * (torch.sum(K_tilde_terms) + torch.sum(trace_terms) +
                       torch.sum(torch.log(noise)) + b * np.log(2 * np.pi) +
                       torch.sum(precision * (y - mean_vector) ** 2))
        
        return L_H, KL_term

    def approximate_posterior(self, index_points, w_l, phi_l):
        # parameters for q_S
        b = index_points.shape[0]

        K_pp = self.kernel(self.inducing_index_points, self.inducing_index_points) # (p,p)
        K_pp_inv = torch.linalg.inv(add_diagonal_jitter(K_pp, self.jitter)) # (p,p)

        K_xx = self.kernel.forward_1d(index_points, index_points) # (x,)
        K_xp = self.kernel(index_points, self.inducing_index_points) # (x,p)
        K_px = torch.transpose(K_xp, 0, 1) # (p,x)

        K_np = self.kernel(index_points, self.inducing_index_points) # (N,p)
        K_pn = torch.transpose(K_np, 0, 1) # (p,N)

        # approximate the parameters for stochastic inducing points
        sigma_l = K_pp + (self.N_train / b) * torch.matmul(K_pn, K_np / phi_l[:,None])
        sigma_l_inv = torch.linalg.inv(add_diagonal_jitter(sigma_l, self.jitter))

        mu_hat = (self.N_train / b) * torch.matmul(torch.matmul(K_pp, torch.matmul(sigma_l_inv, K_pn)), w_l / phi_l)
        A_hat = torch.matmul(K_pp, torch.matmul(sigma_l_inv, K_pp))

        # evaluate the posterior distribution of GP regression
        mean_vector = (self.N_train / b) * torch.matmul(K_xp, torch.matmul(sigma_l_inv, torch.matmul(K_pn, w_l / phi_l))) 
        K_xp_Sigma_l_K_px = torch.matmul(K_xp, torch.matmul(sigma_l_inv, K_px))
        B = K_xx + torch.diagonal(-torch.matmul(K_xp, torch.matmul(K_pp_inv, K_px)) + K_xp_Sigma_l_K_px)

        return mean_vector, B, mu_hat, A_hat


    def forward(self, x, y, raw_y, size_factors, num_samples=1):
        self.train()
        b = y.shape[0]

        # encoder
        qnet_mu, qnet_var = self.encoder(y)
        gp_mu = qnet_mu[:, 0:self.latent_GP_dim]
        gp_var = qnet_var[:, 0:self.latent_GP_dim]

        gaussian_mu = qnet_mu[:, self.latent_GP_dim:]
        gaussian_var = qnet_var[:, self.latent_GP_dim:]

        inside_elbo_recon, inside_elbo_kl = [], []
        gp_p_m, gp_p_v = [], []

        for l in range(self.latent_GP_dim):
            g_p_m_l, g_p_v_l, mu_hat_l, A_hat_l = self.approximate_posterior(x, gp_mu[:, l], gp_var[:, l])
            inside_elbo_recon_l, inside_elbo_kl_l = self.cal_L_H(x=x, y=gp_mu[:,l], noise=gp_var[:,l], mu_hat=mu_hat_l, A_hat=A_hat_l)

            inside_elbo_recon.append(inside_elbo_recon_l)
            inside_elbo_kl.append(inside_elbo_kl_l)
            gp_p_m.append(g_p_m_l)
            gp_p_v.append(g_p_v_l)

        inside_elbo_recon = torch.stack(inside_elbo_recon, dim=-1)
        inside_elbo_kl = torch.stack(inside_elbo_kl, dim=-1)
        inside_elbo_recon = torch.sum(inside_elbo_recon)
        inside_elbo_kl = torch.sum(inside_elbo_kl)

        inside_elbo = inside_elbo_recon - (b / self.N_train) * inside_elbo_kl

        gp_p_m = torch.stack(gp_p_m, dim=1)
        gp_p_v = torch.stack(gp_p_v, dim=1)

        # cross entropy term
        gp_ce_term = gauss_cross_entropy(gp_p_m, gp_p_v, gp_mu, gp_var, device=self.device)
        gp_ce_term = torch.sum(gp_ce_term)

        # KL term of GP prior
        gp_KL_term = gp_ce_term - inside_elbo

        # KL term of Gaussian prior
        gaussian_prior_dist = Normal(torch.zeros_like(gaussian_mu), torch.ones_like(gaussian_var))
        gaussian_post_dist = Normal(gaussian_mu, torch.sqrt(gaussian_var))
        gaussian_KL_term = kl_divergence(gaussian_post_dist, gaussian_prior_dist).sum()

        # decoder
        p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
        p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
        latent_dist = Normal(p_m, torch.sqrt(p_v))
        latent_samples = []
        mean_samples = []
        disp_samples = []
        for _ in range(num_samples):
            latent_samples_ = latent_dist.rsample()
            latent_samples.append(latent_samples_)

        # reconstruction loss
        recon_loss = 0
        for f in latent_samples:
            hidden_samples = self.decoder(f)
            mean_samples_ = self.recon_mean(hidden_samples)
            disp_samples_ = (torch.exp(torch.clamp(self.recon_covariance, -15., 15.))).unsqueeze(0)

            mean_samples.append(mean_samples_)
            disp_samples.append(disp_samples_)
            recon_loss += self.nbloss(x=raw_y, mean=mean_samples_, disp=disp_samples_, scale_factor=size_factors)
        recon_loss = recon_loss / num_samples

        elbo = recon_loss + self.beta * (gp_KL_term + gaussian_KL_term)

        return elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, gp_p_m, gp_p_v, qnet_mu, qnet_var, \
            mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples


    def train_model(self, pos, ncounts, raw_counts, size_factors, lr=0.001, weight_decay=0.001, batch_size=512, num_samples=1, 
            train_ratio=0.95, maxiter=5000, patience=200):
        self.train()

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(ncounts, dtype=self.dtype), 
                        torch.tensor(raw_counts, dtype=self.dtype), torch.tensor(size_factors, dtype=self.dtype))
        
        if train_ratio < 1:
            train_dataset, val_dataset = random_split(dataset=dataset, lengths=[train_ratio, 1.-train_ratio])
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            train_dataset = dataset

        if ncounts.shape[0] * train_ratio > batch_size:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        early_stopping = EarlyStopping(patience=patience, modelfile=self.checkpoints_path)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        KL_queue = deque(maxlen=10)

        print("Training")

        for epoch in range(maxiter):
            elbo_val = 0
            recon_loss_val = 0
            gp_KL_term_val = 0
            gaussian_KL_term_val = 0
            noise_reg_val = 0
            num = 0
            for batch_idx, (x_batch, y_batch, y_raw_batch, sf_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_raw_batch = y_raw_batch.to(self.device)
                sf_batch = sf_batch.to(self.device)

                elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, p_m, p_v, qnet_mu, qnet_var, \
                    mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples= \
                    self.forward(x=x_batch, y=y_batch, raw_y=y_raw_batch, size_factors=sf_batch, num_samples=num_samples)

                self.zero_grad()
                elbo.backward(retain_graph=True)
                optimizer.step()

                elbo_val += elbo.item()
                recon_loss_val += recon_loss.item()
                gp_KL_term_val += gp_KL_term.item()
                gaussian_KL_term_val += gaussian_KL_term.item()

                num += x_batch.shape[0]

                # tune beta
                KL_val = (gp_KL_term.item() + gaussian_KL_term.item()) / x_batch.shape[0]
                KL_queue.append(KL_val)
                avg_KL = np.mean(KL_queue)
                self.beta, _ = self.PID.pid(self.desired_KL_loss*(self.latent_GP_dim+self.latent_Gau_dim), avg_KL)
                if len(KL_queue) >= 10:
                    KL_queue.popleft()


            elbo_val = elbo_val/num
            recon_loss_val = recon_loss_val/num
            gp_KL_term_val = gp_KL_term_val/num
            gaussian_KL_term_val = gaussian_KL_term_val/num
            noise_reg_val = noise_reg_val/num

            print("-"*100)
            print('Training epoch {}, ELBO:{:.8f}, NB loss:{:.8f}, GP KLD loss:{:.8f}, Gaussian KLD loss:{:.8f}'.format(epoch+1, elbo_val, recon_loss_val, gp_KL_term_val, gaussian_KL_term_val))
            print('Current beta', self.beta)
            print('Current kernel scale', torch.clamp(F.softplus(self.kernel.scale), min=1e-10, max=1e4).data)

            if train_ratio < 1:
                validate_elbo_val = 0
                validate_num = 0
                for _, (validate_x_batch, validate_y_batch, validate_y_raw_batch, validate_sf_batch) in enumerate(val_dataloader):
                    validate_x_batch = validate_x_batch.to(self.device)
                    validate_y_batch = validate_y_batch.to(self.device)
                    validate_y_raw_batch = validate_y_raw_batch.to(self.device)
                    validate_sf_batch = validate_sf_batch.to(self.device)

                    validate_elbo, _, _, _, _, _, _, _, _, _, _, _, _, _, _= \
                        self.forward(x=validate_x_batch, y=validate_y_batch, raw_y=validate_y_raw_batch, size_factors=validate_sf_batch, num_samples=num_samples)

                    validate_elbo_val += validate_elbo.item()
                    validate_num += validate_x_batch.shape[0]

                validate_elbo_val = validate_elbo_val / validate_num

                print("Training epoch {}, validating ELBO:{:.8f}".format(epoch+1, validate_elbo_val))
                early_stopping(validate_elbo_val, self)
                if early_stopping.early_stop:
                    print('EarlyStopping: run {} iteration'.format(epoch+1))
                    break

    def sample_latent_embedding(self, X, Y, batch_size=512):
        self.eval()

        X = torch.tensor(X, dtype=self.dtype)
        Y = torch.tensor(Y, dtype=self.dtype)

        latent_samples = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            ybatch = Y[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)

            qnet_mu, qnet_var = self.encoder(ybatch)

            gp_mu = qnet_mu[:, 0:self.latent_GP_dim]
            gp_var = qnet_var[:, 0:self.latent_GP_dim]

            gaussian_mu = qnet_mu[:, self.latent_GP_dim:]
#            gaussian_var = qnet_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.latent_GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.approximate_posterior(xbatch, gp_mu[:, l], gp_var[:, l])
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            latent_samples.append(p_m.data.cpu().detach())

        latent_samples = torch.cat(latent_samples, dim=0)

        return latent_samples.numpy()
