import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 28, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(28, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, embedding_dim, kernel_size=1, stride=1)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        latents = x.permute(0, 2, 3, 1)
        return latents

class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.deconv1 = nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = nn.functional.interpolate(x, size=(28, 28), mode='bilinear', align_corners=True)
        x_recon = torch.sigmoid(x)
        return x_recon

class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.encoder = Encoder(num_embeddings, embedding_dim)
        self.decoder = Decoder(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        latents = self.encoder(x)
        quantized_latents, indices = self.vector_quantize(latents)
        x_recon = self.decoder(quantized_latents)
        return x_recon, indices

    def vector_quantize(self, latents):
        latents_shape = latents.shape
        flat_latents = latents.reshape(-1, self.embedding_dim)
        distances = torch.cdist(flat_latents, self.embedding.weight)
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices)
        quantized = quantized.view(latents_shape)
        encoding_indices = encoding_indices.view(latents_shape[:-1])
        return quantized, encoding_indices

def vq_vae_loss(x_recon, x, beta, encoding_indices, latents):
    recon_loss = F.mse_loss(x_recon, x)
    commit_loss = torch.mean((latents.detach() - encoding_indices.unsqueeze(-1))**2)
    loss = recon_loss + beta * commit_loss
    return loss

num_epochs = 10
batch_size = 128
learning_rate = 1e-3
num_embeddings = 512
embedding_dim = 32
beta = 0.25

train_dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE(num_embeddings, embedding_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        x_recon, encoding_indices = model(images.to(device))
        latents = model.encoder(images.to(device))
        loss = vq_vae_loss(x_recon, images.to(device), beta, encoding_indices, latents)
        loss.backward()
        optimizer.step()
    if i % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(train_loader)}] Loss: {loss.item()}")
        


test_dataset = MNIST(root='data', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

reconstruction_loss = 0
with torch.no_grad():
    for images, _ in test_loader:
        x_recon, _ = model(images.to(device))
        reconstruction_loss += F.mse_loss(x_recon, images.to(device), reduction='sum')     
        reconstruction_loss /= len(test_dataset)
        print(f"Test Set Reconstruction Loss: {reconstruction_loss.item()}")
