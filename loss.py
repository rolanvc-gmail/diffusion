from diffusion_forward import forward_diffusion_sample
from torch.nn import functional as F


def get_loss(model, x_0, t, device, p_sqrt_alpha_cumprod, p_sqrt_one_minus_alpha_cumprod):
    x_noisy, noise = forward_diffusion_sample(x_0, t, p_sqrt_alpha_cumprod, p_sqrt_one_minus_alpha_cumprod, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)