import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import copy

# --- Hyperparameters ---
T = 200
BATCH_SIZE = 64
IMG_SIZE = 28
EMBED_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Cosine Beta Schedule ---
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# --- Noise Schedule ---
# betas = torch.linspace(1e-4, 0.02, T).to(DEVICE)
betas = cosine_beta_schedule(T).to(DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


# --- Dataset ---
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- Timestep Embedding ---
def get_timestep_embedding(timesteps, dim):
    half_dim = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half_dim).float() / half_dim).to(DEVICE)
    angles = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

# --- Forward Diffusion ---
def q_sample(x0, t, noise):
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t.to('cpu')]).view(-1, 1, 1, 1).to(DEVICE)
    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t.to('cpu')]).view(-1, 1, 1, 1).to(DEVICE)
    return sqrt_alphas_cumprod * x0 + sqrt_one_minus * noise

# --- Mini U-Net ---
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.norm2 = nn.GroupNorm(1, out_ch)

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.norm1(self.conv1(x))
        h += self.time_mlp(t)[:, :, None, None]
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        return h + self.res_conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, time_emb_dim=EMBED_DIM):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.enc1 = ResBlock(1, 32, time_emb_dim)
        self.enc2 = ResBlock(32, 64, time_emb_dim)
        self.middle = ResBlock(64, 64, time_emb_dim)
        self.dec1 = ResBlock(64, 32, time_emb_dim)
        self.dec2 = ResBlock(32, 1, time_emb_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x1 = self.enc1(x, t_emb)
        x2 = self.enc2(F.avg_pool2d(x1, 2), t_emb)
        x3 = self.middle(x2, t_emb)
        x4 = F.interpolate(self.dec1(x3, t_emb), scale_factor=2, mode='nearest')
        return self.dec2(x4 + x1, t_emb)
    
class UNet_128_cond(nn.Module):
    def __init__(self, time_emb_dim=EMBED_DIM, num_classes=10):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)
        
        self.enc1 = ResBlock(1, 64, time_emb_dim)
        self.enc2 = ResBlock(64, 128, time_emb_dim)
        self.middle = ResBlock(128, 128, time_emb_dim)
        self.dec1 = ResBlock(128, 64, time_emb_dim)
        self.dec2 = ResBlock(64, 1, time_emb_dim)

    def forward(self, x, t, y):
        t_emb = self.time_mlp(t)
        y_emb = self.label_emb(y)
        cond_emb = t_emb + y_emb
        
        x1 = self.enc1(x, cond_emb)
        x2 = self.enc2(F.avg_pool2d(x1, 2), cond_emb)
        x3 = self.middle(x2, cond_emb)
        x4 = F.interpolate(self.dec1(x3, cond_emb), scale_factor=2, mode='nearest')
        return self.dec2(x4 + x1, cond_emb)
    
model = UNet_128_cond().to(DEVICE)
# model.load_state_dict(torch.load(r"diff_weights/cond_model_epoch_100.pt"))

def sample(model, n_samples=1, y=None):
    model.eval()
    x = torch.randn(n_samples, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)

    if y is None:
        y = torch.randint(0, 10, (n_samples,), device=DEVICE)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=DEVICE, dtype=torch.long)
        t_emb = get_timestep_embedding(t_batch, EMBED_DIM)

        with torch.no_grad():
            eps_theta = model(x, t_emb, y)

        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        beta_t = betas[t]

        mu = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta)

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = mu + torch.sqrt(beta_t) * noise

    return x