import random
import os
from torchvision.utils import save_image
import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import argparse
from progress.bar import IncrementalBar

from dataset import Cityscapes, Facades, Maps
from dataset import transforms as T
from gan.generator import UnetGenerator
from gan.discriminator import ConditionalDiscriminator
from gan.criterion import GeneratorLoss, DiscriminatorLoss
from gan.utils import Logger, initialize_weights

parser = argparse.ArgumentParser(prog = 'top', description='Train Pix2Pix')
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="facades", help="Name of the dataset: ['facades', 'maps', 'cityscapes']")
parser.add_argument("--batch_size", type=int, default=1, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
args = parser.parse_args()

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = T.Compose([T.Resize((256,256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])])
# models
print('Defining models!')
generator = UnetGenerator().to(device)
discriminator = ConditionalDiscriminator().to(device)
# optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
# loss functions
g_criterion = GeneratorLoss(alpha=100)
d_criterion = DiscriminatorLoss()
# dataset
print(f'Downloading "{args.dataset.upper()}" dataset!')
if args.dataset=='cityscapes':
    dataset = Cityscapes(root='.', transform=transforms, download=True, mode='train')
elif args.dataset=='maps':
    dataset = Maps(root='.', transform=transforms, download=True, mode='train')
else:
    dataset = Facades(root='.', transform=transforms, download=True, mode='train')
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
print('Start of training process!')
logger = Logger(filename=args.dataset)
# Create a directory for saving samples
os.makedirs(f'samples/{args.dataset}', exist_ok=True)

for epoch in range(args.epochs):
    ge_loss = 0.
    de_loss = 0.
    start = time.time()
    bar = IncrementalBar(f'[Epoch {epoch+1}/{args.epochs}]', max=len(dataloader))
    for x, real in dataloader:
        x = x.to(device)
        real = real.to(device)

        # Generator`s loss
        fake = generator(x)
        fake_pred = discriminator(fake, x)
        g_loss = g_criterion(fake, real, fake_pred)

        # Discriminator`s loss
        fake = generator(x).detach()
        fake_pred = discriminator(fake, x)
        real_pred = discriminator(real, x)
        d_loss = d_criterion(fake_pred, real_pred)

        # Generator`s params update
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Discriminator`s params update
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Add batch losses
        ge_loss += g_loss.item()
        de_loss += d_loss.item()
        bar.next()
    bar.finish()
    
    # Obtain per epoch losses
    g_loss = ge_loss / len(dataloader)
    d_loss = de_loss / len(dataloader)
    # Count timeframe
    end = time.time()
    tm = (end - start)
    logger.add_scalar('generator_loss', g_loss, epoch+1)
    logger.add_scalar('discriminator_loss', d_loss, epoch+1)
    logger.save_weights(generator.state_dict(), 'generator')
    logger.save_weights(discriminator.state_dict(), 'discriminator')
    
    # Inference and save a random sample
    generator.eval()  # Set generator to evaluation mode
    with torch.no_grad():
        # Pick a random sample from the dataset
        sample_idx = random.randint(0, len(dataset) - 1)
        x_sample, real_sample = dataset[sample_idx]
        x_sample = x_sample.unsqueeze(0).to(device)  # Add batch dimension
        real_sample = real_sample.unsqueeze(0).to(device)
        
        # Generate fake image
        fake_sample = generator(x_sample)
        
        # Save real, input, and generated images for comparison
        save_image((x_sample + 1) / 2, f'samples/{args.dataset}/input_{epoch+1}.png')  # Input
        save_image((real_sample + 1) / 2, f'samples/{args.dataset}/real_{epoch+1}.png')  # Ground Truth
        save_image((fake_sample + 1) / 2, f'samples/{args.dataset}/fake_{epoch+1}.png')  # Generated
    
    generator.train()  # Set generator back to training mode
    
    print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] ETA: %.3fs" % (epoch+1, args.epochs, g_loss, d_loss, tm))

logger.close()
print('End of training process!')
