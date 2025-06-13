import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Encoder(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super().__init__()

        # Kết hợp input và condition
        combined_dim = input_dim + condition_dim

        # Mạng encoder
        self.encoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Các lớp để tính mean và log variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c):
        # Kết hợp input và condition
        combined_input = torch.cat([x, c], dim=1)

        # Encode
        hidden = self.encoder(combined_input)

        # Tính mean và log variance
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, output_dim):
        super().__init__()

        # Kết hợp latent vector và condition
        combined_dim = latent_dim + condition_dim

        # Mạng decoder
        self.decoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z, c):
        # Kết hợp latent vector và condition
        combined_input = torch.cat([z, c], dim=1)

        # Decode
        output = self.decoder(combined_input)
        return output


class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        # Khởi tạo encoder và decoder
        self.encoder = Encoder(input_dim, condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick để lấy mẫu từ N(mu, var) = N(0,1) * var + mu"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        # Encode
        mu, logvar = self.encoder(x, c)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_x = self.decoder(z, c)

        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """Tính ELBO loss = Reconstruction + KL Divergence"""

        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Tổng loss
        return recon_loss + kl_div

    def sample(self, n_samples, c):
        """Tạo mẫu mới từ prior p(z) với condition c"""
        device = next(self.parameters()).device

        # Lấy mẫu từ prior N(0,1)
        z = torch.randn(n_samples, self.latent_dim).to(device)

        # Decode với condition
        samples = self.decoder(z, c)

        return samples


# Ví dụ sử dụng:
def train_cvae(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, c) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            recon_x, mu, logvar = model(x, c)

            # Tính loss
            loss = model.loss_function(recon_x, x, mu, logvar)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}')


# Khởi tạo model
input_dim = 784  # Ví dụ với MNIST
condition_dim = 10  # Số class
hidden_dim = 400
latent_dim = 20

model = CVAE(input_dim, condition_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Giả sử train_loader đã được chuẩn bị
# train_cvae(model, train_loader, optimizer, epochs=50)