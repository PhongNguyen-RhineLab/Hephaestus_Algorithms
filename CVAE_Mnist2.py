import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime

def show_reconstruction(x, x_recon, num_images=10, save_path=None):
    try:
        x = x.view(-1, 28, 28).detach().cpu().numpy()
        x_recon = x_recon.view(-1, 28, 28).detach().cpu().numpy()
        num_images = min(num_images, x.shape[0], x_recon.shape[0])
        if num_images == 0:
            print(f"Warning: No images to visualize (x.shape: {x.shape}, x_recon.shape: {x_recon.shape})")
            return
        plt.figure(figsize=(num_images * 2, 4))
        for i in range(num_images):
            plt.subplot(2, num_images, i + 1)
            plt.imshow(x[i], cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title("Original")
            plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(x_recon[i], cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title("Reconstructed")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved image to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Error in show_reconstruction: {e}")

def prepare_cluster_batch(images, labels, k_max=10, n_max=50):
    D = images.size(1)
    k = torch.unique(labels).tolist()
    k_count = len(k)
    cluster_map = {label: [] for label in k}
    for img, lbl in zip(images, labels):
        cluster_map[lbl.item()].append(img)
    centroids = []
    clusters = [[]]
    for label in k:
        points = torch.stack(cluster_map[label])
        centroid = points.mean(dim=0)
        centroids.append(centroid)
        clusters[0].append(points)
    centroids = torch.stack(centroids).unsqueeze(0)
    k_tensor = torch.tensor([k_count])
    return centroids, clusters, k_tensor

def process_cluster_data(centroids, cluster_points, k, k_max=10, n_max=50, D=784):
    batch_size = centroids.size(0)
    x_padded = F.pad(centroids, (0, 0, 0, k_max - centroids.size(1)))
    x_tensor = x_padded.view(batch_size, -1)
    cluster_data = torch.zeros(batch_size, k_max, n_max, D)
    for b in range(batch_size):
        for i in range(k[b]):
            points = cluster_points[b][i]
            n_i = min(points.size(0), n_max)
            cluster_data[b, i, :n_i, :] = points[:n_i]
    cluster_flat = cluster_data.view(batch_size, -1)
    k_mask = torch.zeros(batch_size, k_max)
    for b in range(batch_size):
        k_mask[b, :k[b]] = 1.0
    k_real = k.float().unsqueeze(1) / k_max
    c_tensor = torch.cat([cluster_flat, k_mask, k_real], dim=1)
    return x_tensor, c_tensor

def k_medians_cost(images, labels, centroids, k, k_max=10):
    """Calculate k-medians clustering cost based on the paper's definition."""
    device = images.device
    cost = 0.0
    valid_centroids = centroids.view(-1, k_max, 784)[:, :k.item()].reshape(-1, 784)
    if valid_centroids.size(0) == 0:
        print(f"Warning: No valid centroids for batch, k={k.item()}")
        return 0.0
    for img in images:
        distances = torch.norm(valid_centroids - img.unsqueeze(0), dim=1, p=2)
        cost += torch.min(distances)
    return cost.item()

def compute_total_k_medians_cost(data_loader, model, k_max, n_max, D, device):
    """Compute total k-medians cost over the entire dataset."""
    model.eval()
    total_cost = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            centroids, cluster_points, k = prepare_cluster_batch(images, labels, k_max, n_max)
            x_tensor, c_tensor = process_cluster_data(centroids, cluster_points, k, k_max, n_max, D)
            output = model(x_tensor.to(device), c_tensor.to(device))
            cost = k_medians_cost(images, labels, output['x_recon'], k, k_max)
            total_cost += cost
    model.train()
    return total_cost

def compute_ground_truth_cost(data_loader, k_max=10, device='cuda'):
    """Compute k-medians cost using ground-truth centroids (mean of each class)."""
    cluster_sums = {i: torch.zeros(784, device=device) for i in range(k_max)}
    cluster_counts = {i: 0 for i in range(k_max)}
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        for img, lbl in zip(images, labels):
            lbl = lbl.item()
            if lbl < k_max:
                cluster_sums[lbl] += img
                cluster_counts[lbl] += 1
    centroids = torch.stack([cluster_sums[i] / max(cluster_counts[i], 1) for i in range(k_max)])
    cost = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            for img in images:
                distances = torch.norm(centroids - img.unsqueeze(0), dim=1, p=2)
                cost += torch.min(distances).item()
    return cost

class CVAE_Encoder(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, x, c):
        xc = torch.cat([x, c], dim=1)
        h = F.relu(self.fc1(xc))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min=-10, max=10)
        return mu, logvar

class CVAE_Prior(nn.Module):
    def __init__(self, cond_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, c):
        h = F.relu(self.fc1(c))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min=-10, max=10)
        return mu, logvar

class CVAE_Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)
        h = F.relu(self.fc1(zc))
        return torch.sigmoid(self.fc_out(h))

class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = CVAE_Encoder(input_dim, cond_dim, hidden_dim, latent_dim)
        self.decoder = CVAE_Decoder(latent_dim, cond_dim, hidden_dim, input_dim)
        self.prior = CVAE_Prior(cond_dim, hidden_dim, latent_dim)

    def forward(self, x, c):
        mu_post, logvar_post = self.encoder(x, c)
        std = torch.exp(0.5 * logvar_post)
        z = mu_post + std * torch.randn_like(std)
        x_recon = self.decoder(z, c)
        mu_prior, logvar_prior = self.prior(c)
        return {
            'x_recon': x_recon,
            'mu': mu_post,
            'logvar': logvar_post,
            'mu_prior': mu_prior,
            'logvar_prior': logvar_prior,
            'z': z,
        }

def cvae_loss(x, output, beta=0.5):
    x_recon = output['x_recon']
    mu = output['mu']
    logvar = output['logvar']
    mu_prior = output['mu_prior']
    logvar_prior = output['logvar_prior']
    if torch.isnan(x_recon).any() or torch.isnan(x).any():
        print("NaN detected in x_recon or x")
    if torch.isnan(mu).any() or torch.isnan(logvar).any():
        print("NaN detected in mu or logvar")
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl = -0.5 * torch.sum(
        1 + (logvar - logvar_prior) - ((mu - mu_prior) ** 2 + logvar.exp()) / (logvar_prior.exp() + 1e-10)
    ) / x.size(0)
    return recon_loss + beta * kl, recon_loss, kl

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(mnist_train, batch_size=256, shuffle=True)

# Configuration
k_max = 10
n_max = 50
D = 784
input_dim = k_max * D
cond_dim = k_max * n_max * D + k_max + 1
hidden_dim = 512
latent_dim = 64

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CVAE(input_dim=input_dim, cond_dim=cond_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)  # Increased learning rate
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # Adjusted schedule

# Create output directory
save_dir = r'D:\Coding Project\Hephaestus_Algorithms\reconstructions'
os.makedirs(save_dir, exist_ok=True)
print(f"Saving images to: {save_dir}")

# Initialize CSV logging
csv_path = os.path.normpath(os.path.join(save_dir, f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'))
results = []

# Compute ground-truth cost once
ground_truth_cost = compute_ground_truth_cost(train_loader, k_max, device)
print(f"Ground-truth k-medians cost: {ground_truth_cost:.4f}")

# Training loop
for epoch in range(100):  # Increased epochs
    model.train()
    total_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    k_medians_cost_total = 0
    count = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        centroids, cluster_points, k = prepare_cluster_batch(images, labels, k_max, n_max)
        x_tensor, c_tensor = process_cluster_data(centroids, cluster_points, k, k_max, n_max, D)
        x_tensor, c_tensor = x_tensor.to(device), c_tensor.to(device)
        output = model(x_tensor, c_tensor)
        loss, recon_loss, kl_loss = cvae_loss(x_tensor, output, beta=0.5)
        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch+1}, batch {count+1}")
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        k_medians_cost_value = k_medians_cost(images, labels, output['x_recon'], k, k_max)
        k_medians_cost_total += k_medians_cost_value
        total_loss += loss.item()
        recon_loss_total += recon_loss.item()
        kl_loss_total += kl_loss.item()
        count += 1
    scheduler.step()
    # Compute total dataset cost
    total_k_medians_cost = compute_total_k_medians_cost(train_loader, model, k_max, n_max, D, device)
    # Log centroid norms for debugging
    with torch.no_grad():
        centroid_norms = torch.norm(output['x_recon'].view(-1, k_max, 784)[:, :k.item()], dim=2)
        print(f"[Epoch {epoch+1}] x_tensor shape: {x_tensor.shape}, x_recon shape: {output['x_recon'].shape}, k: {k.item()}, Centroid norms: {centroid_norms}")
    num_images = min(k.item(), k_max)
    if num_images > 0:
        original_images = x_tensor.view(-1, k_max, 784)[:, :num_images].reshape(-1, 784)[:num_images]
        reconstructed_images = output['x_recon'].view(-1, k_max, 784)[:, :num_images].reshape(-1, 784)[:num_images]
        save_path = os.path.normpath(os.path.join(save_dir, f'reconstruction_epoch_{epoch + 1}.png'))
        show_reconstruction(original_images, reconstructed_images, num_images=num_images, save_path=save_path)
    else:
        print(f"[Epoch {epoch+1}] Skipping visualization: no clusters to display")
    # Log results
    epoch_results = {
        'Epoch': epoch + 1,
        'Total_Loss': total_loss / count if count > 0 else 0.0,
        'Recon_Loss': recon_loss_total / count if count > 0 else 0.0,
        'KL_Loss': kl_loss_total / count if count > 0 else 0.0,
        'k_medians_Cost_Per_Batch': k_medians_cost_total / count if count > 0 else 0.0,
        'k_medians_Cost_Total': total_k_medians_cost,
        'Ground_Truth_Cost': ground_truth_cost
    }
    results.append(epoch_results)
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)
    print(f"[Epoch {epoch+1}] Loss: {total_loss / count:.4f}, Recon: {recon_loss_total / count:.4f}, KL: {kl_loss_total / count:.4f}, "
          f"k-medians Cost (Per Batch): {k_medians_cost_total / count:.4f}, k-medians Cost (Total): {total_k_medians_cost:.4f}, "
          f"Ground-truth Cost: {ground_truth_cost:.4f}, Saved to {csv_path}")