import torch
from diffusion_forward import get_index_from_list
from test_fwd_process import betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, T
from simple_unet import SimpleUnet
from diffusion_data import IMG_SIZE, BATCH_SIZE, show_tensor_image
import matplotlib.pyplot as plt
from simple_unet import SimpleUnet


@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, device, epoch):
    """

    Parameters
    ----------
    model
    device
    epoch

    Returns
    -------

    """
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)  # img is noise of size img_size,img_size
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10  # num_images of what?
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:  # i steps backwards from T to 0
        t = torch.full((1,), i, device=device, dtype=torch.long)  # t is a tensor of shape(1,) filled with value i
        img = sample_timestep(model, img, t)  # img is noise, t is the timestep

        # While we go through the timesteps from  T to 0
        # we plot every stepsize from T to 0.
        if i % stepsize == 0:
            plt.subplot(1, num_images, i // stepsize + 1)
            show_tensor_image(img.detach().cpu())  # img is Tensor(1,3, 64, 64)
    plt.savefig(F"images/image_{epoch}.png")
