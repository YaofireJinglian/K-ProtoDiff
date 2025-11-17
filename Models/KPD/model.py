import math
import torch
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.KPD.transformer import Transformer
from Models.KPD.model_utils import default, identity, extract
from Layer.KBasisNet import KBasisNet
from Layer.cross_attention import channel_AutoCorrelationLayer
from typing import List
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import torch.fft

class FDM(nn.Module):
    def __init__(self, f_l_ratio=0.33, f_h_ratio=0.66):
        super(FDM, self).__init__()
        self.f_l_ratio = f_l_ratio
        self.f_h_ratio = f_h_ratio

    def forward(self, x):
        B, T, D = x.shape
        X_freq = fft.fft(x, dim=1)
        f_l = int(T * self.f_l_ratio)
        f_h = int(T * self.f_h_ratio)

        low_freq = torch.zeros_like(X_freq)
        low_freq[:, :f_l, :] = X_freq[:, :f_l, :]
        F_l = fft.ifft(low_freq, dim=1).real

        mid_freq = torch.zeros_like(X_freq)
        mid_freq[:, f_l:f_h, :] = X_freq[:, f_l:f_h, :]
        F_m = fft.ifft(mid_freq, dim=1).real

        high_freq = torch.zeros_like(X_freq)
        high_freq[:, f_h:, :] = X_freq[:, f_h:, :]
        F_h = fft.ifft(high_freq, dim=1).real

        return F_l, F_m, F_h


class TaylorKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, Q=4, P=3):
        super(TaylorKAN, self).__init__()
        self.Q = Q
        self.P = P
        self.phi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(P, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(Q)
        ])
        self.psi = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                ) for _ in range(P)
            ]) for _ in range(Q)
        ])

    def forward(self, x):
        B, T, D = x.shape
        outputs = []
        for q in range(self.Q):
            psi_outs = []
            for p in range(self.P):
                psi_out = self.psi[q][p](x)
                psi_outs.append(psi_out)
            psi_concat = torch.cat(psi_outs, dim=-1)
            phi_out = self.phi[q](psi_concat)
            outputs.append(phi_out)
        return torch.sum(torch.cat(outputs, dim=-1), dim=-1, keepdim=True)


class MOKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, Q=4, P=3):
        super(MOKAN, self).__init__()
        self.kan_l = TaylorKAN(input_dim, hidden_dim, Q, P)
        self.kan_m = TaylorKAN(input_dim, hidden_dim, Q, P)
        self.kan_h = TaylorKAN(input_dim, hidden_dim, Q, P)

    def forward(self, F_l, F_m, F_h):
        H_l = self.kan_l(F_l)
        H_m = self.kan_m(F_m)
        H_h = self.kan_h(F_h)
        H = torch.cat([H_l, H_m, H_h], dim=-1)
        return H


class KPA(nn.Module):
    def __init__(self, d_model, num_proto=8):
        super(KPA, self).__init__()
        self.d_model = d_model
        self.num_proto = num_proto
        self.P = nn.Parameter(torch.randn(num_proto, d_model))
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)

    def forward(self, H):
        Q = self.W_Q(H)
        K = self.W_K(self.P)
        attn = torch.softmax(Q @ K.T / (self.d_model ** 0.5), dim=-1)
        Z_p = attn @ self.P
        return Z_p, attn


class KPL(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, d_model=64, Q=4, P=3, num_proto=8,
                 f_l_ratio=0.33, f_h_ratio=0.66):
        super(KPL, self).__init__()
        self.fdm = FDM(f_l_ratio=f_l_ratio, f_h_ratio=f_h_ratio)
        self.mokan = MOKAN(input_dim, hidden_dim, Q, P)
        self.kpa = KPA(d_model, num_proto)
        self.proj = nn.Linear(hidden_dim * 3, d_model)

    def forward(self, X_0):
        F_l, F_m, F_h = self.fdm(X_0)
        H = self.mokan(F_l, F_m, F_h)
        H_proj = self.proj(H)
        Z_p, attn = self.kpa(H_proj)
        return Z_p, attn
class KPD(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            use_ff=True,
            reg_weight=None,
            kpl_hidden_dim=64,
            kpl_Q=4,
            kpl_P=3,
            kpl_num_proto=8,
            kpl_f_l_ratio=0.33,
            kpl_f_h_ratio=0.66,
            **kwargs
    ):
        super(KPD, self).__init__()
        self.d_model = d_model
        self.epsilon = 1E-5
        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.loss_type = loss_type

        # Transformer backbone
        self.model = Transformer(
            n_feat=feature_size,
            n_channel=seq_length,
            n_layer_enc=n_layer_enc,
            n_layer_dec=n_layer_dec,
            n_heads=n_heads,
            attn_pdrop=attn_pd,
            resid_pdrop=resid_pd,
            mlp_hidden_times=mlp_hidden_times,
            max_len=seq_length,
            n_embd=d_model,
            conv_params=[kernel_size, padding_size],
            **kwargs
        )


        self.kpl = KPL(
            input_dim=feature_size,
            hidden_dim=kpl_hidden_dim,
            d_model=d_model,
            Q=kpl_Q,
            P=kpl_P,
            num_proto=kpl_num_proto,
            f_l_ratio=kpl_f_l_ratio,
            f_h_ratio=kpl_f_h_ratio
        )

        # Beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.fast_sampling = self.sampling_timesteps < timesteps

        # Helper to register buffers
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('timesteps', torch.arange(timesteps, 0, -1))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def output(self, x, x_mark, t, padding_masks=None):

        low_freq, mid_freq, high_freq = fdm(x)
        kan_block = MultiOrderKANBlock(input_dim=D)
        H = kan_block(low_freq, mid_freq, high_freq)  # (B, T, D)
        kpa = PrototypeAssignment(d_model=D, num_prototypes=K)
        Z_p, A = kpa(H)
        model_output = self.model(Z_p, t, padding_masks=padding_masks)  # transformer
        return model_output

    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None, guidance_scale=None):
        """Modified to support guidance scale in predictions"""
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        x_start = self.output(x, None, t, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        
        # Add guidance scale support if provided
        if guidance_scale is not None and hasattr(self, 'unconditional_output'):
            uncond_x_start = self.unconditional_output(x, None, t, padding_masks)
            uncond_x_start = maybe_clip(uncond_x_start)
            x_start = uncond_x_start + guidance_scale * (x_start - uncond_x_start)
            pred_noise = uncond_pred_noise + guidance_scale * (pred_noise - uncond_pred_noise)
            
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True, guidance_scale=None):
        """Modified to pass guidance_scale to model_predictions"""
        _, x_start = self.model_predictions(x, t, clip_denoised, guidance_scale=guidance_scale)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, clip_denoised=True, cond_fn=None, model_kwargs=None,
                T_max=1, inv_guidance_scale=0.0, lambda_step=None):
        """Main function with z-sampling improvements"""
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        
        # Set default lambda_step if not provided
        if lambda_step is None:
            lambda_step = 0.01 * len(self.timesteps)
        # print(self.timesteps)
        # Original forward process
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.
        
        if cond_fn is not None:
            model_mean = self.condition_mean(
                cond_fn, model_mean, model_log_variance, x, t=batched_times, 
                model_kwargs=model_kwargs
            )
        
        pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
        
        # R-Sampling optimization for early timesteps
        if t < lambda_step:
            for _ in range(T_max):
                #  inverse step
           
                inv_mean, inv_var, inv_log_var, _ = self.p_mean_variance(
                    x=pred_series, 
                    t=batched_times+1 if t < len(self.timesteps)-1 else batched_times,
                    clip_denoised=clip_denoised,
                    guidance_scale=inv_guidance_scale
                )
                inv_noise = torch.randn_like(pred_series) if t > 0 else 0.
                inv_pred_series = inv_mean + (0.5 * inv_log_var).exp() * inv_noise
                
                # forward step
                model_mean, _, model_log_variance, x_start = \
                    self.p_mean_variance(x=inv_pred_series, t=batched_times, clip_denoised=clip_denoised)
                noise = torch.randn_like(inv_pred_series) if t > 0 else 0.
                pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
                
                if cond_fn is not None:
                    pred_series = self.condition_mean(
                        cond_fn, pred_series, model_log_variance, 
                        inv_pred_series, t=batched_times, model_kwargs=model_kwargs
                    )
        
        return pred_series, x_start


    @torch.no_grad()
    def sample(self, shape):
        device = self.betas.device
        ts = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            ts, _ = self.p_sample(ts, t)
        return ts

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        ts = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(ts, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                ts = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(ts)
            ts = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return ts
    
    def generate_mts(self, batch_size=16, model_kwargs=None, cond_fn=None):
        feature_size, seq_length = self.feature_size, self.seq_length
        if cond_fn is not None:
            sample_fn = self.fast_sample_cond if self.fast_sampling else self.sample_cond
            return sample_fn((batch_size, seq_length, feature_size), model_kwargs=model_kwargs, cond_fn=cond_fn)
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn((batch_size, seq_length, feature_size))

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, t, x_mark=None, target=None, noise=None, padding_masks=None):
        noise = default(noise, lambda: torch.randn_like(x_start))  # (B, L, C)
        if target is None:
            target = x_start

        B, L, C = x_start.shape
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # (B, L, C)
        model_out = self.output(x_noisy, x_fused, t, padding_masks)  # (B, L, C)
        train_loss = self.loss_fn(model_out, target, reduction='none')  # (B, L, C)
        fourier_loss = torch.tensor([0.], device=x_start.device)
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')\
                           + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss +=  self.ff_weight * fourier_loss
        
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def forward(self, x, **kwargs):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        # KPL
        Z_p, attn = self.kpl(x_start)  # Z_p: (B, L, d_model), attn: (B, L, num_proto)
        x_fused = torch.cat([x_start, Z_p], dim=-1)  # (B, L, C + d_model)
        fusion_proj = nn.Linear(C + self.kpl.kpa.d_model, self.d_model).to(x_start.device)
        x_fused = fusion_proj(x_fused)  # (B, L, d_model)
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x_fused, t=t, **kwargs)

    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def fast_sample_infill(self, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True, model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        ts = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(ts, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                ts = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(ts)

            ts = pred_mean + sigma * noise
            ts = self.langevin_fn(sample=ts, mean=pred_mean, sigma=sigma, t=time_cond,
                                   tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
            target_t = self.q_sample(target, t=time_cond)
            ts[partial_mask] = target_t[partial_mask]

        ts[partial_mask] = target[partial_mask]

        return ts

    def sample_infill(
        self,
        shape, 
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):

        batch, device = shape[0], self.betas.device
        ts = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            ts = self.p_sample_infill(x=ts, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)
        
        ts[partial_mask] = target[partial_mask]
        return ts
    
    def p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_ts = model_mean + sigma * noise

        pred_ts = self.langevin_fn(sample=pred_ts, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
        
        target_t = self.q_sample(target, t=batched_times)
        pred_ts[partial_mask] = target_t[partial_mask]

        return pred_ts

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t,
        coef_=0.
    ):
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()
            
                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter((input_embs_param.data + coef_ * sigma.mean().item() * epsilon).detach())

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample
    
    def condition_mean(self, cond_fn, mean, log_variance, x, t, model_kwargs=None):

        gradient = cond_fn(x=x, t=t, **model_kwargs)
        new_mean = (
            mean.float() + torch.exp(log_variance) * gradient.float()
        )
        return new_mean
    
    def condition_score(self, cond_fn, x_start, x, t, model_kwargs=None):

        alpha_bar = extract(self.alphas_cumprod, t, x.shape)

        eps = self.predict_noise_from_start(x, t, x_start)
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        pred_xstart = self.predict_start_from_noise(x, t, eps)
        model_mean, _, _ = self.q_posterior(x_start=pred_xstart, x_t=x, t=t)
        return model_mean, pred_xstart
    
    def sample_cond(
        self,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        cond_fn=None
    ):

        batch, device = shape[0], self.betas.device
        ts = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            ts, x_start = self.p_sample(ts, t, clip_denoised=clip_denoised, cond_fn=cond_fn,
                                         model_kwargs=model_kwargs)
        return ts

    def fast_sample_cond(
        self,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        cond_fn=None
    ):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        ts = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(ts, time_cond, clip_x_start=clip_denoised)

            if cond_fn is not None:
                _, x_start = self.condition_score(cond_fn, x_start, ts, time_cond, model_kwargs=model_kwargs)
                pred_noise = self.predict_noise_from_start(ts, time_cond, x_start)

            if time_next < 0:
                ts = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(ts)
            ts = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return ts




class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],  # No features for YearEnd
        offsets.QuarterEnd: [MonthOfYear],  # Only MonthOfYear
        offsets.MonthEnd: [MonthOfYear],  # Only MonthOfYear
        offsets.Week: [DayOfMonth, WeekOfYear],  # DayOfMonth, WeekOfYear
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],  # DayOfWeek, DayOfMonth, DayOfYear
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],  # DayOfWeek, DayOfMonth, DayOfYear
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],  # HourOfDay, DayOfWeek, DayOfMonth, DayOfYear
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth],  # MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth
        offsets.Second: [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek],  # SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0, 0.999)
