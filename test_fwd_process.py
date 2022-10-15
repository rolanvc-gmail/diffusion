import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from diffusion_forward import linear_beta_schedule, forward_diffusion_sample
from diffusion_data import load_transformed_dataset, show_tensor_image, BATCH_SIZE, IMG_SIZE


T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1.-betas  # alphas is as long as betas
alphas_cumprod = torch.cumprod(alphas, axis=0)  # is as long as alphas, each element is the cumulative product of previous elements.
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # almost identical to alphas_cumprod, but first element is a "1.0"
sqrt_recip_alphas = torch.sqrt(1.0/alphas)   # as long as alphas, but each element is the square root of the reciprocal.
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # as long as alphas, each element is the sqrt of the cumprod.
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)  # as long as alphas.
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # as long as alphas


def main():
    data = load_transformed_dataset()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Simulate forward diffusion
    image = next(iter(dataloader))[0]
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    step_size = int(T / num_images)

    for idx in range(0, T, step_size):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images + 1, int(idx / step_size) + 1)
        image, noise = forward_diffusion_sample(image, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        show_tensor_image(image)

    plt.show()


if __name__ == "__main__":
    main()
