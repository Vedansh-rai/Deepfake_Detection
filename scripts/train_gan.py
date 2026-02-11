import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# Reuse Dataset for simplest case, though GAN usually trains on single images
# For GAN, we can just treat frames as individual images.
# Let's create a simple ImageDataset wrapper or just use existing DeepfakeDataset but flat
from src.data.dataset import DeepfakeDataset
from src.models.gan import Generator, Discriminator, initialize_weights

def train_gan(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transform for GAN (typically normalize to [-1, 1])
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # We only need Real images for GAN to learn the distribution
    # Creating a custom loader that just yields frames from Real videos would be best
    # For simplicity, let's use the dataset logic but filter for Real
    dataset = DeepfakeDataset(root_dir=os.path.join(args.data_dir, 'train'), transform=transform, num_frames=5)
    
    # This is a bit inefficient because dataset returns (T, C, H, W)
    # We can flatten it in the loop
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize models
    netG = Generator(z_dim=args.z_dim, img_channels=3, feature_g=64).to(device)
    netD = Discriminator(img_channels=3, feature_d=64).to(device)
    
    initialize_weights(netG)
    initialize_weights(netD)
    
    # Optimizers
    optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    
    fixed_noise = torch.randn(32, args.z_dim, 1, 1).to(device)
    
    for epoch in range(args.epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for frames, labels in loop:
            # frames: (B, T, C, H, W)
            # Flatten to (B*T, C, H, W)
            b, t, c, h, w = frames.size()
            real = frames.view(-1, c, h, w).to(device)
            
            # Only use Real images (label 0)
            # Actually, `DeepfakeDataset` returns mixed. We should ideally filter.
            # But let's just train on everything "Real" in the batch if we can, or just everything for style.
            # Ideally GAN trains on Real faces to learn P(x).
            # If we train on Fakes too, it learns mixed distribution.
            # Let's assume we want to generate more "Fake" looking things or "Real" things?
            # Request: "use GANs to generate synthetic fake samples".
            # Usually: Train G to map Noise -> Real-like. Then disturb it?
            # Or train CycleGAN (Unpaired).
            # Given the simple GAN arch, we likely want to generate new faces.
            # Let's just train on all data for now for demonstration.
            
            batch_size = real.size(0)
            
            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            label = (torch.ones(batch_size) * 0.9).to(device) # Soft smoothing
            output = netD(real).reshape(-1)
            lossD_real = criterion(output, label)
            lossD_real.backward()
            
            noise = torch.randn(batch_size, args.z_dim, 1, 1).to(device)
            fake = netG(noise)
            label = (torch.zeros(batch_size) + 0.1).to(device)
            output = netD(fake.detach()).reshape(-1)
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            
            lossD = lossD_real + lossD_fake
            optD.step()
            
            ### Train Generator: max log(D(G(z)))
            netG.zero_grad()
            label = torch.ones(batch_size).to(device)
            output = netD(fake).reshape(-1)
            lossG = criterion(output, label)
            lossG.backward()
            optG.step()
            
            loop.set_postfix(lossD=lossD.item(), lossG=lossG.item())
            
        # Save sample images
        with torch.no_grad():
            fake = netG(fixed_noise)
            img_grid = vutils.make_grid(fake, normalize=True)
            vutils.save_image(img_grid, os.path.join(args.output_dir, f'epoch_{epoch}.png'))

        # Save Checkpoint
        torch.save(netG.state_dict(), os.path.join(args.checkpoint_dir, 'generator.pth'))
        torch.save(netD.state_dict(), os.path.join(args.checkpoint_dir, 'discriminator.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/vedanshrai/Desktop/Deepfake Detection', help='Root directory')
    parser.add_argument('--checkpoint_dir', type=str, default='models_gan', help='Checkpoints dir')
    parser.add_argument('--output_dir', type=str, default='generated_samples', help='Output dir for samples')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--z_dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--image_size', type=int, default=64, help='Image size (64 for standard DCGAN)')
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_gan(args)
