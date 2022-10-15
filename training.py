import torch
from simple_unet import SimpleUnet
from torch.optim import Adam
from diffusion_data import BATCH_SIZE
from loss import get_loss
from sampling import sample_plot_image
from diffusion_data import load_transformed_dataset

T = 1000
model = SimpleUnet()
dataloader = load_transformed_dataset()


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100 # Try more!

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      loss = get_loss(model, batch[0], t)
      loss.backward()
      optimizer.step()

      if epoch % 5 == 0 and step == 0:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        sample_plot_image()