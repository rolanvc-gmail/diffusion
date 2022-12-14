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
import matplotlib.pyplot as plt
import time


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(F"device is: {device}")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 100  # Try more!

    start_time = time.time()
    for epoch in range(epochs):
        if epoch > 1:
            running_time = time.time()
            ave_per_epoch = (running_time - start_time)/epoch
            print(f"ave time per epoch is: {ave_per_epoch} seconds.")

        print(F"epoch:{epoch}")

        step_start_time = time.time()
        for step, batch in enumerate(dataloader):  # Total dataset size is 16185. Batch size is 128. Expect total steps to be 16185/128 = 125
            print(f"epoch:{epoch}, step:{step}")
            if step > 0:
                step_running_time = time.time()
                ave_per_step = (step_running_time - start_time) / step
                print(f"ave time per step is: {ave_per_step} seconds.")
            optimizer.zero_grad()
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model=model, x_0=batch[0], t=t, p_sqrt_alpha_cumprod=sqrt_alphas_cumprod,
                            p_sqrt_one_minus_alpha_cumprod=sqrt_one_minus_alphas_cumprod, device=device)
            loss.backward()
            optimizer.step()

            if step == 0:  # epoch % 5 == 0 and step == 0:  # Let's plot every epoch, but only for the first batch of each epoch...
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                sample_plot_image(model, betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance, device, epoch, T)


if __name__ == "__main__":
    main()
