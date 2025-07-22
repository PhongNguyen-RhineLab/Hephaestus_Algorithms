import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

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
                plt.title("Gốc")

            plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(x_recon[i], cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title("Tái tạo")

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

# --- Encoder ---
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

# --- Prior ---
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

# --- Decoder ---
class CVAE_Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)
        h = F.relu(self.fc1(zc))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return torch.sigmoid(self.fc_out(h))

# --- CVAE Full Model ---
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

# --- Loss ---
def cvae_loss(x, output, cluster_points, k, k_max=10, n_max=50, D=784, beta=0.01):
    """
    Compute CVAE loss with k-medians clustering cost.

    Args:
        x (torch.Tensor): Input centroids [batch_size, k_max * D]
        output (dict): Model output containing 'x_recon', 'mu', 'logvar', 'mu_prior', 'logvar_prior'
        cluster_points (list): List of cluster points per batch [batch_size, list of k tensors of shape [n_i, D]]
        k (torch.Tensor): Number of clusters per batch [batch_size]
        k_max (int): Maximum number of clusters
        n_max (int): Maximum number of points per cluster
        D (int): Dimension of each point (784 for MNIST)
        beta (float): Weight for KL divergence

    Returns:
        tuple: (total_loss, recon_loss, kl_loss)
    """
    x_recon = output['x_recon']  # [batch_size, k_max * D]
    mu = output['mu']
    logvar = output['logvar']
    mu_prior = output['mu_prior']
    logvar_prior = output['logvar_prior']

    # Check for NaN/Inf
    if torch.isnan(x_recon).any() or torch.isnan(x).any():
        print("NaN detected in x_recon or x")
    if torch.isinf(x_recon).any() or torch.isinf(x).any():
        print("Inf detected in x_recon or x")
    if torch.isnan(mu).any() or torch.isnan(logvar).any():
        print("NaN detected in mu or logvar")

    batch_size = x.size(0)

    # Reshape reconstructed centroids to [batch_size, k_max, D]
    x_recon = x_recon.view(batch_size, k_max, D)

    # Compute k-medians clustering cost
    total_cost = 0.0
    total_points = 0

    for b in range(batch_size):
        k_b = k[b].item()  # Number of clusters in this batch
        if k_b == 0:
            continue  # Skip empty batches

        for i in range(k_b):
            points = cluster_points[b][i]  # [n_i, D]
            n_i = min(points.size(0), n_max)
            if n_i == 0:
                continue  # Skip empty clusters

            # Get reconstructed centroid for cluster i
            centroid_recon = x_recon[b, i, :]  # [D]

            # Compute unsquared Euclidean distances to reconstructed centroid
            distances = torch.norm(points[:n_i] - centroid_recon, dim=1, p=2)

            # k-medians: sum of unsquared distances
            cluster_cost = torch.sum(distances)

            total_cost += cluster_cost
            total_points += n_i

    # Average cost over all points
    recon_loss = total_cost / max(total_points, 1) if total_points > 0 else torch.tensor(0.0, device=x.device)

    # KL Divergence
    kl = -0.5 * torch.sum(
        1 + (logvar - logvar_prior) - ((mu - mu_prior) ** 2 + torch.exp(logvar)) / (torch.exp(logvar_prior) + 1e-5)
    ) / x.size(0)

    # Total loss
    total_loss = recon_loss + beta * kl

    return total_loss, recon_loss, kl

# --- Load MNIST ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1) / 255.0)
])

mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(mnist_train, batch_size=256, shuffle=True)

# Configuration
k_max = 10
n_max = 50
D = 784
input_dim = k_max * D
cond_dim = k_max * n_max * D + k_max + 1
hidden_dim = 1024
latent_dim = 512  # Increased to 512

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CVAE(input_dim=input_dim, cond_dim=cond_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Create output directory
save_dir = r'D:\Coding Project\Hephaestus_Algorithms\reconstructions'
os.makedirs(save_dir, exist_ok=True)
print(f"Saving images to: {save_dir}")

# Early stopping
best_recon_loss = float('inf')
patience = 10
trigger_times = 0

# Training loop
# Inside the training loop
for epoch in range(100):
    model.train()
    total_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    count = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        centroids, cluster_points, k = prepare_cluster_batch(images, labels, k_max=k_max, n_max=n_max)
        x_tensor, c_tensor = process_cluster_data(centroids, cluster_points, k, k_max=k_max, n_max=n_max, D=D)
        x_tensor, c_tensor = x_tensor.to(device), c_tensor.to(device)

        output = model(x_tensor, c_tensor)
        # Use k-medians loss
        loss, recon_loss, kl_loss = cvae_loss(
            x_tensor, output, cluster_points, k, k_max=k_max, n_max=n_max, D=D, beta=0.01
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf loss detected at epoch {epoch+1}, batch {count+1}")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        recon_loss_total += recon_loss.item()
        kl_loss_total += kl_loss.item()
        count += 1

    scheduler.step(recon_loss_total / count)  # Adjust LR based on recon loss

    avg_recon_loss = recon_loss_total / count
    if avg_recon_loss < best_recon_loss:
        best_recon_loss = avg_recon_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    with torch.no_grad():
        print(f"[Epoch {epoch+1}] x_tensor shape: {x_tensor.shape}, x_recon shape: {output['x_recon'].shape}, k: {k.item()}")
        num_images = min(k.item(), k_max)
        if num_images > 0:
            original_images = x_tensor.view(-1, k_max, 784)[:, :num_images].reshape(-1, 784)[:num_images]
            reconstructed_images = output['x_recon'].view(-1, k_max, 784)[:, :num_images].reshape(-1, 784)[:num_images]
            save_path = os.path.normpath(os.path.join(save_dir, f'reconstruction_epoch_{epoch + 1}.png'))
            show_reconstruction(original_images, reconstructed_images, num_images=num_images, save_path=save_path)
        else:
            print(f"[Epoch {epoch+1}] Skipping visualization: no clusters to display")

    print(f"[Epoch {epoch+1}] Loss: {total_loss / count:.4f}, Recon: {recon_loss_total / count:.4f}, KL: {kl_loss_total / count:.4f}")