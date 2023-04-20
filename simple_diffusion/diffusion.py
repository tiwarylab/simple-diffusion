import torch 
from functorch import vmap
import torch.nn.functional as F
import torch.nn as nn

def polynomial_noise(t, alpha_max, alpha_min, s=1e-5):
    """
    Same schedule used in Hoogeboom et. al. (Equivariant Diffusion for Molecule Generation in 3D)
    """
    T = t[-1]
    alphas = (1-2*s)*(1-(t/T)**2) + s
    a = alphas[1:]/alphas[:-1]
    a[a**2 < 0.001] = 0.001
    alpha_schedule = torch.cumprod(a, 0)
    return alpha_schedule

NOISE_FUNCS = {
    "polynomial": polynomial_noise,
              }

class VPDiffusion:
    """
    Performs a diffusion according to the VP-SDE.
    """
    def __init__(self,
                 num_diffusion_timesteps,
                 noise_schedule="polynomial",
                 alpha_max=20.,
                 alpha_min=0.01,
                 NOISE_FUNCS=NOISE_FUNCS,
                ):

        self.bmul = vmap(torch.mul)
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.times = torch.arange(num_diffusion_timesteps)
        self.alphas = NOISE_FUNCS[noise_schedule](torch.arange(num_diffusion_timesteps+1),
                                                  alpha_max,
                                                  alpha_min)
        
    def get_alphas(self):
        return self.alphas

    def forward_kernel(self, x0, t):
        """
        Maginal transtion kernels of the forward process. p(x_t|x_0).
        """
        alphas_t = self.alphas[t]
        noise = torch.randn_like(x0)
        # interpolate between data and noise
        x_t = self.bmul(x0, alphas_t.sqrt()) + self.bmul(noise, (1-alphas_t).sqrt())
        return x_t, noise

    def reverse_kernel(self, x_t, t, backbone, pred_type):
        """
        Marginal transition kernels of the reverse process. q(x_0|x_t).
        """
        # get noise schedule
        alphas_t = self.alphas[t]
        
        # predict noise added to data
        if pred_type == "noise":
            noise = backbone(x_t, alphas_t)
            noise_interp = self.bmul(noise, (1-alphas_t).sqrt())
            # predict x0 given x_t and noise
            x0_t = self.bmul((x_t - noise_interp), 1/alphas_t.sqrt())
            
        # predict data
        elif pred_type == "x0":
            x0_t = backbone(x_t, alphas_t)
            x0_interp = self.bmul(x0_t, (alphas_t).sqrt())
            # predict noise given x_t and x0
            noise = self.bmul((x_t - x0_interp), 1/(1-alphas_t).sqrt())
        else:
            raise Exception("Please provide a valid prediction type: 'noise' or 'x0'")

        return x0_t, noise

    def reverse_step(self, x_t, t, t_next, backbone, pred_type):
        """
        Stepwise transition kernel of the reverse process q(x_t-1|x_t).
        """

        # getting noise schedule
        alphas_t = self.alphas[t]
        alphas_t_next = self.alphas[t_next]
        
        # computing x_0' ~ p(x_0|x_t)
        x0_t, noise = self.reverse_kernel(x_t, t, backbone, pred_type)
        
        # computing x_t+1 = f(x_0', x_t, noise)
        xt_next = self.bmul(alphas_t_next.sqrt(), x0_t) + self.bmul((1-alphas_t_next).sqrt(), noise)
        return xt_next

    def sample_prior(self, xt):
        """
        Generates a sample from a prior distribution z ~ p(z).
        """
        noise = torch.randn_like(xt)
        return noise