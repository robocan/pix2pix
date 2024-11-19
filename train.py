import random
import os
from torchvision.utils import save_image

# Create a directory for saving samples
os.makedirs(f'samples/data', exist_ok=True)

for epoch in range(epochs):
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
