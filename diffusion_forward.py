import torch
import torch.functional as F


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """ Returns a specific index t of  passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,)*(len(x_shape)-1))).to(t.device)


def forward_diffusion_sample(x_0, t, p_sqrt_alpha_cumprod, p_sqrt_one_minus_alpha_cumprod, device="mps"):
    """ Takes an image and timestep as input
    and returns a noisy version of it.
    """
    noise = torch.randn_like(x_0)  # create noise like x_0
    sqrt_alphas_cumprod_t = get_index_from_list(p_sqrt_alpha_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(p_sqrt_one_minus_alpha_cumprod, t, x_0.shape)
    print(F"t size: {t.shape[0]}   sqrt_one_minus_alpha size:{sqrt_one_minus_alphas_cumprod_t.shape[0]} sqrt_alphas_cumprod_t size: {sqrt_one_minus_alphas_cumprod_t.shape[0]} noise size:{noise.shape[0]}")
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

