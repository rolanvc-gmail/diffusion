import torch
from simple_unet import SimpleUnet
from torch.optim import Adam
from diffusion_data import BATCH_SIZE
from loss import get_loss
from sampling import sample_plot_image
from diffusion_data import load_transformed_dataset
from diffusion_forward import linear_beta_schedule
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
def main():
    T = 1000  # Timesteps
    betas = linear_beta_schedule(timesteps=T)

    # Pre-calculate different terms for closed form
    alphas = 1.-betas  # alphas is as long as betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)  # is as long as alphas, each element is the cumulative product of previous elements.
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # almost identical to alphas_cumprod, but first element is a "1.0"
    sqrt_recip_alphas = torch.sqrt(1.0/alphas)   # as long as alphas, but each element is the square root of the reciprocal.
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # as long as alphas, each element is the sqrt of the cumprod.
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)  # as long as alphas.
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # as long as alphas

    model = SimpleUnet()
    data = load_transformed_dataset()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    device = "mps"
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 100  # Try more!

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model=model, x_0=batch[0], t=t, p_sqrt_alpha_cumprod=sqrt_alphas_cumprod,
                            p_sqrt_one_minus_alpha_cumprod=sqrt_one_minus_alphas_cumprod, device=device)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                sample_plot_image()


if __name__ == "__main__":
    main()
