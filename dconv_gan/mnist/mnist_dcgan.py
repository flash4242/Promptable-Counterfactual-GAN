#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# Configuration dictionary
config = {
    'cuda': True,
    'data_path': '/mnt/data',
    'batch_size': 128,
    'image_channel': 1,
    'z_dim': 100,
    'g_hidden': 64,
    'd_hidden': 64,
    'x_dim': 64,
    'epochs': 20,
    'real_label': 1.,
    'fake_label': 0.,
    'lr': 2e-4,
    'seed': 1,
    'out_dir': os.getcwd(),
}

# Setup
torch.manual_seed(config['seed'])
config['cuda'] = config['cuda'] and torch.cuda.is_available()
device = torch.device("cuda:0" if config['cuda'] else "cpu")
print(f"Using device: {device}")
if config['cuda']:
    torch.cuda.manual_seed(config['seed'])
    print(f"CUDA version: {torch.version.cuda}")

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize(config['x_dim']),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = dset.MNIST(
    root=config['data_path'],
    train=True,
    download=False,
    transform=transform
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=2
)

# Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(config['z_dim'], config['g_hidden'] * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config['g_hidden'] * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(config['g_hidden'] * 8, config['g_hidden'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config['g_hidden'] * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(config['g_hidden'] * 4, config['g_hidden'] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config['g_hidden'] * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(config['g_hidden'] * 2, config['g_hidden'], 4, 2, 1, bias=False),
            nn.BatchNorm2d(config['g_hidden']),
            nn.ReLU(True),
            nn.ConvTranspose2d(config['g_hidden'], config['image_channel'], 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(config['image_channel'], config['d_hidden'], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config['d_hidden'], config['d_hidden'] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config['d_hidden'] * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config['d_hidden'] * 2, config['d_hidden'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config['d_hidden'] * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config['d_hidden'] * 4, config['d_hidden'] * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config['d_hidden'] * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config['d_hidden'] * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# Initialize models
netG = Generator().to(device)
netG.apply(weights_init)
netD = Discriminator().to(device)
netD.apply(weights_init)

# Loss and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=config['lr'], betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=config['lr'], betas=(0.5, 0.999))

# Fixed noise for progress visualization
viz_noise = torch.randn(config['batch_size'], config['z_dim'], 1, 1, device=device)

# Training
img_list = []
G_losses, D_losses = [], []
iters = 0

print("Starting Training Loop...")
for epoch in range(config['epochs']):
    for i, data in enumerate(dataloader):
        ############################
        # (1) Update D network
        ###########################
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), config['real_label'], dtype=torch.float, device=device)
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, config['z_dim'], 1, 1, device=device)
        fake = netG(noise)
        label.fill_(config['fake_label'])
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        label.fill_(config['real_label'])  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Logging
        if i % 200 == 0:
            print(f"[{epoch}/{config['epochs']}][{i}/{len(dataloader)}] "
                  f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                  f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if iters % 500 == 0 or (epoch == config['epochs'] - 1 and i == len(dataloader) - 1):
            with torch.no_grad():
                fake = netG(viz_noise).detach().cpu()
            img = vutils.make_grid(fake, padding=2, normalize=True)
            img_list.append(img)

        iters += 1

# Save loss plot
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(config['out_dir'], 'loss_plot.png'))

# Save real vs fake images
real_batch = next(iter(dataloader))

plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=True), (1,2,0)))

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1,2,0)))

plt.tight_layout()
plt.savefig(os.path.join(config['out_dir'], 'real_vs_fake.png'))
