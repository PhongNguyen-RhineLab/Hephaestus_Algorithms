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
from sklearn.decomposition import PCA
from datetime import datetime

def show_reconstruction(x, x_recon, num_images=10, save_path=None):
    # Disable visualization for PCA features (not images)
    print("Skipping visualization (PCA features, not images).")
    return

# def show_reconstruction(x, x_recon, num_images=10, save_path=None):
#     try:
#         x = x.view(-1, 28, 28).detach().cpu().numpy()
#         x_recon = x_recon.view(-1, 28, 28).detach().cpu().numpy()
#         num_images = min(num_images, x.shape[0], x_recon.shape[0])
#         if num_images == 0:
#             print(f"Warning: No images to visualize (x.shape: {x.shape}, x_recon.shape: {x_recon.shape})")
#             return
#         plt.figure(figsize=(num_images * 2, 4))
#         for i in range(num_images):
#             plt.subplot(2, num_images, i + 1)
#             plt.imshow(x[i], cmap='gray')
#             plt.axis('off')
#             if i == 0:
#                 plt.title("Original")
#             plt.subplot(2, num_images, i + 1 + num_images)
#             plt.imshow(x_recon[i], cmap='gray')
#             plt.axis('off')
#             if i == 0:
#                 plt.title("Reconstructed")
#         plt.tight_layout()
#         if save_path:
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             plt.savefig(save_path, bbox_inches='tight')
#             print(f"Saved image to {save_path}")
#         plt.close()
#     except Exception as e:
#         print(f"Error in show_reconstruction: {e}")

def prepare_cluster_batch(images, labels, k_max=10, n_max=50):
    """
    Prepare centroids and cluster points per sample in a minibatch.
    Returns:
        centroids: (B, k_max, D)
        clusters: list of length B, each is a list of tensors (points for each cluster)
        k: (B,) tensor of number of clusters per sample
    """
    B, D = images.shape  # images: (B, D) already flattened
    centroids = []
    clusters = []
    k_list = []

    for b in range(B):
        lbls = labels[b].unsqueeze(0)  # single label
        img = images[b]

        # since MNIST labels are single-class, we just make one cluster per image
        # but if labels[b] is actually a vector of cluster assignments, handle multiple
        unique_labels = torch.unique(lbls)
        k_count = len(unique_labels)
        k_list.append(k_count)

        centroids_b = []
        clusters_b = []
        for ul in unique_labels:
            points = img.unsqueeze(0)  # (1, D) since each sample has one label
            centroid = points.median(dim=0).values  # (D,)
            centroids_b.append(centroid)
            clusters_b.append(points)

        # pad centroids if less than k_max
        while len(centroids_b) < k_max:
            centroids_b.append(torch.zeros(D, device=images.device))
            clusters_b.append(torch.zeros(0, D, device=images.device))

        centroids.append(torch.stack(centroids_b[:k_max]))
        clusters.append(clusters_b[:k_max])

    centroids = torch.stack(centroids)  # (B, k_max, D)
    k_tensor = torch.tensor(k_list, device=images.device)
    return centroids, clusters, k_tensor


def process_cluster_data(centroids, cluster_points, k, k_max=10, n_max=50, D=784):
    """
    Convert centroids + cluster data into tensors for CVAE.
    centroids: (B, k_max, D)
    cluster_points: list of length B, each with k_max entries of (n_i, D)
    k: (B,)
    """
    B = centroids.size(0)

    # Flatten centroids
    x_tensor = centroids.view(B, -1)  # (B, k_max*D)

    # Create padded cluster data
    cluster_data = torch.zeros(B, k_max, n_max, D, device=centroids.device)
    for b in range(B):
        for i in range(k[b]):
            points = cluster_points[b][i]
            n_i = min(points.size(0), n_max)
            if n_i > 0:
                cluster_data[b, i, :n_i, :] = points[:n_i]

    cluster_flat = cluster_data.view(B, -1)  # (B, k_max*n_max*D)

    # Cluster mask + normalized cluster count
    k_mask = torch.zeros(B, k_max, device=centroids.device)
    for b in range(B):
        k_mask[b, :k[b]] = 1.0
    k_real = k.float().unsqueeze(1) / k_max  # (B,1)

    # Final condition tensor
    c_tensor = torch.cat([cluster_flat, k_mask, k_real], dim=1)  # (B, cond_dim)

    return x_tensor, c_tensor


def k_medians_cost(images, labels, centroids, k, k_max=10):
    """
    Calculate k-medians clustering cost for a minibatch.
    Args:
        images: (B, D)
        labels: (B,)
        centroids: (B, k_max*D) or (B, k_max, D)
        k: (B,)
    Returns:
        cost: scalar float
    """
    device = images.device
    B, D = images.shape

    # If centroids are flattened, reshape them
    if centroids.dim() == 2:
        centroids = centroids.view(B, k_max, D)

    total_cost = 0.0

    for b in range(B):
        k_b = k[b].item()
        if k_b == 0:
            continue

        valid_centroids = centroids[b, :k_b]  # (k_b, D)
        img = images[b]                       # (D,)

        # Euclidean distance from img to all centroids
        distances = torch.norm(valid_centroids - img.unsqueeze(0), dim=1, p=2)
        total_cost += torch.min(distances).item()
    return total_cost


def compute_total_k_medians_cost(data_loader, model, k_max, n_max, D, device):
    """
    Compute total k-medians cost over the entire dataset.
    Uses model reconstructions as centroids.
    """
    model.eval()
    total_cost = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Prepare data per-sample
            centroids, cluster_points, k = prepare_cluster_batch(images, labels, k_max, n_max)
            x_tensor, c_tensor = process_cluster_data(centroids, cluster_points, k, k_max, n_max, D)

            # Forward pass
            output = model(x_tensor.to(device), c_tensor.to(device))

            # Compute cost per batch
            batch_cost = k_medians_cost(images, labels, output['x_recon'], k, k_max)
            total_cost += batch_cost

    model.train()
    return total_cost


def compute_ground_truth_cost(data_loader, k_max, device, D=64):
    # Use dynamic dimensionality instead of hardcoding 784
    cluster_sums = [torch.zeros(D).to(device) for _ in range(k_max)]
    cluster_counts = [0 for _ in range(k_max)]
    all_labels = []
    all_images = []

    for batch in data_loader:
        images, labels = batch
        images = images.to(device)
        all_labels.extend(labels.tolist())
        all_images.append(images)

        for img, lbl in zip(images, labels):
            cluster_sums[lbl] += img
            cluster_counts[lbl] += 1

    # Compute cluster centers
    cluster_means = []
    for s, c in zip(cluster_sums, cluster_counts):
        if c > 0:
            cluster_means.append(s / c)
        else:
            cluster_means.append(torch.zeros(D).to(device))

    # Stack
    all_images = torch.cat(all_images, dim=0)
    cluster_means = torch.stack(cluster_means)

    # Compute ground-truth clustering cost (Euclidean distance to mean)
    cost = 0.0
    for img, lbl in zip(all_images, all_labels):
        diff = img - cluster_means[lbl]
        cost += torch.norm(diff, p=2).item()

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

# --- Load MNIST ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 → 784
])
mnist_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# --- Subset to 1797 samples ---
m = 1797
X = torch.stack([mnist_full[i][0] for i in range(m)])  # shape [1797, 784]
y = torch.tensor([mnist_full[i][1] for i in range(m)]) # labels

# --- Apply PCA to 64 dimensions ---
pca = PCA(n_components=64)
X_pca = pca.fit_transform(X.numpy())  # shape [1797, 64]

# --- Convert back to tensor ---
X_tensor = torch.tensor(X_pca, dtype=torch.float32)
y_tensor = y

# --- Wrap into DataLoader ---
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

# --- Config (now d=64 instead of 784) ---
k_max = 10
n_max = 50
D = 64   # <<<<<<<<<<<<<< CHANGED

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
ground_truth_cost = compute_ground_truth_cost(train_loader, k_max, device, D)
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
        x_recon_batch = output['x_recon'].view(-1, k_max, D)
        centroid_norms = []
        for b in range(x_recon_batch.size(0)):
            centroid_norms.append(torch.norm(x_recon_batch[b, :k[b]], dim=1).cpu())
        print(f"[Epoch {epoch + 1}] Example centroid norms (first batch entry): {centroid_norms[0]}")
    # Visualize first sample in the batch
    k_first = int(k[0].item())
    num_images = min(k_first, k_max)

    if num_images > 0:
        original_images = x_tensor[0].view(k_max, 64)[:num_images]
        reconstructed_images = output['x_recon'][0].view(k_max, 64)[:num_images]

        save_path = os.path.normpath(os.path.join(save_dir, f'reconstruction_epoch_{epoch + 1}.png'))
        show_reconstruction(original_images, reconstructed_images, num_images=num_images, save_path=save_path)
    else:
        print(f"[Epoch {epoch + 1}] Skipping visualization: no clusters to display")

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