import torch
from types import SimpleNamespace
import numpy as np
import os

from loader import TorsionLoader
from Unet1D import Unet1D
from backbone import ConvBackbone1D
from diffusion import VPDiffusion

def sample_batch(batch_size, loader, diffusion, backbone, pred_type="x0"):

    def sample_prior(batch_size, shape):
        "Generates samples of gaussian noise"
        prior_sample =  torch.randn(batch_size, *shape[1:], dtype=torch.float)
        return prior_sample

    def get_adjacent_times(times):
        """
        Pairs t with t+1 for all times in the time-discretization
        of the diffusion process.
        """
        times_next = torch.cat((torch.Tensor([0]).long(), times[:-1]))
        return list(zip(reversed(times), reversed(times_next)))

    xt = sample_prior(batch_size, loader.data_std.shape)
    time_pairs = get_adjacent_times(diffusion.times)

    for t, t_next in time_pairs:
        print(int(t))
        t = torch.Tensor.repeat(t, batch_size)
        t_next = torch.Tensor.repeat(t_next, batch_size)
        xt_next = diffusion.reverse_step(xt, t, t_next, backbone, pred_type=pred_type)
        xt = xt_next
    return xt

def save_batch(batch, save_prefix, save_idx):
    os.makedirs(directory.sample_path, exist_ok=True)
    file = os.path.join(directory.sample_path, f"{save_prefix}_idx={save_idx}.npz")
    np.savez_compressed(file, data=batch)

def sample_loop(num_samples, batch_size, save_prefix, loader, diffusion, backbone):

    n_runs = max(num_samples//batch_size, 1)
    if num_samples <= batch_size:
        batch_size = num_samples

    with torch.no_grad():
        for save_idx in range(n_runs):
            x0 = sample_batch(batch_size, loader, diffusion, backbone)
            save_batch(x0, save_prefix, save_idx)


directory = SimpleNamespace(model_path="../saved_models/",
                            data_path="../data/aib9.npy",
                            sample_path="../samples/",
                            identifier="aib9"
                           )

loader = TorsionLoader(directory.data_path)

resnet_block_groups = 8
num_torsions = loader.__getitem__(0).shape[-1]
model_dim = int(np.ceil(num_torsions/resnet_block_groups) * resnet_block_groups)


model = Unet1D(dim=model_dim,
               channels=1,
               dim_mults=(1,2,4,8),
               resnet_block_groups=resnet_block_groups,
               learned_sinusoidal_cond=True,
               learned_sinusoidal_dim=16,
               self_condition=True
              )

backbone = ConvBackbone1D(model=model, # model
                          data_shape=num_torsions, # data shape
                          target_shape=model_dim, # network shape
                          num_dims=len(loader.data_std.shape),
                          lr=1e-4
                         )

diffusion = VPDiffusion(num_diffusion_timesteps=100)

backbone.load_model(directory, 10)

sample_loop(50000, 10000, "aib9", loader, diffusion, backbone)
