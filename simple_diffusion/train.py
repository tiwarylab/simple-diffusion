import torch 
from types import SimpleNamespace
import numpy as np
import os

from loader import TorsionLoader
from Unet1D import Unet1D
from backbone import ConvBackbone1D


def train_loop(train_loader, backbone, diffusion, num_epochs=50):
    
    def l2_loss(x, x_pred):
        return (x - x_pred).pow(2).sum((1,2)).pow(0.5).mean()
    
    for epoch in range(num_epochs):
        for i, b in enumerate(train_loader, 0):

            t = torch.randint(low=0, 
                          high=diffusion.num_diffusion_timesteps, 
                          size=(b.size(0),)).long()

            b_t, e_0 = diffusion.forward_kernel(b, t)
            b_0, e_t = diffusion.reverse_kernel(b_t, t, backbone, "x0")

            loss = l2_loss(b_t, b_0)
            backbone.optim.zero_grad()
            loss.backward()
            backbone.optim.step()

            if i % 100 == 0:
                print(f"step: {i}, loss {loss.detach():.3f}")
        backbone.save_state(directory, epoch)
        

directory = SimpleNamespace(model_path="../saved_models/",
                            data_path="../data/aib9.npy",
                            sample_path="../samples/",
                            identifier="aib9"
                           )

loader = TorsionLoader(directory.data_path)

resnet_block_groups = 8 # model size
num_torsions = loader.__getitem__(0).shape[-1]
model_dim = int(np.ceil(num_torsions/resnet_block_groups) * resnet_block_groups)


model = Unet1D(dim=model_dim,
               channels=1,
               resnet_block_groups=resnet_block_groups,
               learned_sinusoidal_cond=True,
               learned_sinusoidal_dim=16
              )

backbone = ConvBackbone1D(model=model, # model
                          data_shape=num_torsions, # data shape
                          target_shape=model_dim, # network shape
                          num_dims=len(loader.data.shape),
                          lr=1e-4
                         )

diffusion = VPDiffusion(num_diffusion_timesteps=100)

train_loader = torch.utils.data.DataLoader(loader, batch_size=512, shuffle=True)

# training the diffusion model
train_loop(train_loader, backbone, diffusion)