import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import time
import numpy as np


class potds_torch(Dataset):
    def __init__(self, selected_pots, transform=None):
        self.selected_pots = selected_pots
        self.transform = transform
    
    def __len__(self):
        return len(self.selected_pots)

    def __getitem__(self,index):
        pot = self.selected_pots[index]

        if self.transform:
            pot = self.transform(pot)
        return pot


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()  
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential( # ImageSize: 256 x 256
            nn.Conv2d(1, 8,  kernel_size = 3, stride=2, padding=1), # ImageSize: 128 x 128
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size = 3, stride=2, padding=1), # ImageSize: 64 x 64
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size = 3, stride=2, padding=1), # ImageSize: 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size = 3, stride=2, padding=1), # ImageSize: 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size = 3, stride=2, padding=1), # ImageSize: 8 x 8
            nn.Flatten(),
            nn.ReLU(True),
            nn.Linear(8*8*128, 128),
            ) 
        
        self.mu = torch.nn.Linear(128, latent_dims)
        self.sigma = torch.nn.Linear(128, latent_dims)
           
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 8 * 8 * 128), 
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(128, 8, 8)), # ImageSize: 8 x 8
            nn.ConvTranspose2d(128, 64, kernel_size = 3, stride=2, padding=1, output_padding=1), # ImageSize: 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size = 3, stride=2, padding=1, output_padding=1), # ImageSize: 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size = 3, stride=2, padding=1, output_padding=1), # ImageSize: 64 x 64
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size= 3, stride=2, padding=1, output_padding=1), # ImageSize: 128 x 128
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size = 3, stride=2, padding=1, output_padding=1), # ImageSize: 256 x 256
            nn.Sigmoid()
            )               
        

        
    def reparameterize_function(self, mu, sigma):
        eps = torch.randn(mu.size(0), mu.size(1)).to(mu.get_device())
        z = mu + eps * torch.exp(sigma/2.) 
        return z
        
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.mu(x), self.sigma(x)
        encoded = self.reparameterize_function(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded



def TrainingVAE(vae, EPOCHS, data_loader_train,  device, loss_fn, optimizer):

    start_time = time.time()
    outputs_train = []
    losses_train = []
    for epoch in range(EPOCHS):
            vae.train()
            for batch_idx, img in enumerate(data_loader_train):
                    
                    img = img.to(device)
                    encoded, z_mean, z_log_var, decoded = vae(img.float())
                    
                    kl_div = -0.5 * torch.sum(1 + z_log_var 
                                        - z_mean**2 
                                        - torch.exp(z_log_var), 
                                        axis=1) # sum over latent dimension

                    batchsize = kl_div.size(0)
                    kl_div = kl_div.mean() # average over batch dimension

                    pixelwise = loss_fn(decoded, img.float(), reduction='none')
                    pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
                    pixelwise = pixelwise.mean() # average over batch dimension

                    loss = pixelwise + kl_div

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                    print(f"Epoch [{epoch+1}/{EPOCHS}] Batch {batch_idx+1}/{len(data_loader_train)}, Loss : {loss.item():.4f}")
            
            outputs_train.append((epoch, img, decoded))
            losses_train.append((epoch, loss.item()))
            loss_train = np.array(losses_train)
    training_time = (time.time() - start_time)

    print(f"Training took {training_time:.3f} seconds")
    return outputs_train, losses_train


def TestingVAE(vae, EPOCHS, data_loader_test,  device, loss_fn):

    outputs_tst = []
    losses_tst = []

    with torch.no_grad():
        for epoch in range(EPOCHS):
            for i, img in enumerate(data_loader_test):
                img = img.to(device)
                encoded, z_mean, z_log_var, decoded = vae(img.float())
                    
                kl_div = -0.5 * torch.sum(1 + z_log_var 
                                        - z_mean**2 
                                        - torch.exp(z_log_var), 
                                        axis=1) # sum over latent dimension

                batchsize = kl_div.size(0)
                kl_div = kl_div.mean() # average over batch dimension

                pixelwise = loss_fn(decoded, img.float(), reduction='none')
                pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
                pixelwise = pixelwise.mean() # average over batch dimension

                loss = pixelwise + kl_div
            outputs_tst.append((epoch, img, decoded))
            losses_tst.append((epoch, loss.item()))
            loss_tst = np.array(losses_tst)
        print(f"Mean test loss: {np.mean(loss_tst[0][1]):.2f}")
        return outputs_tst, losses_tst
