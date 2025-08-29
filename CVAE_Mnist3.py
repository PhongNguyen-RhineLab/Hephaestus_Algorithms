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
import random

# ---------------------------
# Visualization (kept off)
# ---------------------------
def show_reconstruction(x, x_recon, num_images=10, save_path=None):
    print("Skipping visualization (PCA features, not images).")
    return

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

### <<< NEW/CHANGED >>>
def flip_labels(y_true: torch.Tensor, alpha: float, num_classes: int = 10, seed: int = 1234) -> torch.Tensor:
    """
    Simulate a predictor with error rate alpha by randomly flipping labels
    for an alpha fraction of the dataset.
    """
    g = torch.Generator().manual_seed(seed)
    m = y_true.shape[0]
    y_pred = y_true.clone()
    num_flip = int(alpha * m)
    if num_flip == 0:
        return y_pred
    idx = torch.randperm(m, generator=g)[:num_flip]
    for i in idx:
        cur = int(y_pred[i].item())
        # choose a different label
        new_label = cur
        while new_label == cur:
            new_label = torch.randint(low=0, high=num_classes, size=(1,), generator=g).item()
        y_pred[i] = new_label
    return y_pred

### <<< NEW/CHANGED >>>
def batch_predictor_centroids(images: torch.Tensor, labels: torch.Tensor, k_max: int, D: int):
    """
    Compute per-batch centroids for each label in {0..k_max-1} using coordinate-wise *median*.
    Returns:
        centroids_kD: (k_max, D)
        present_mask: (k_max,) 1 if cluster non-empty else 0
        clusters_list: length k_max; each tensor is (n_i, D) of points for cluster i
        k_present: number of non-empty clusters
    """
    centroids = []
    clusters_list = []
    present_mask = torch.zeros(k_max, device=images.device)
    for lbl in range(k_max):
        pts = images[labels == lbl]
        clusters_list.append(pts)
        if pts.shape[0] > 0:
            centroid = pts.median(dim=0).values   # <<< MEDIAN (true k-medians)
            centroids.append(centroid)
            present_mask[lbl] = 1.0
        else:
            centroids.append(torch.zeros(D, device=images.device))
    centroids_kD = torch.stack(centroids, dim=0)
    k_present = int(present_mask.sum().item())
    return centroids_kD, present_mask, clusters_list, k_present

### <<< NEW/CHANGED >>>
def prepare_cluster_batch_from_predictor(images, pred_labels, k_max=10, n_max=50):
    """
    Build CVAE (x, c) from the predictor clusters *at the batch level*.
    All samples in the batch receive the same set of k_max centroids and cluster members
    computed from the batch’s predicted labels.

    Returns:
        x_tensor: (B, k_max*D)  - flattened centroids, same for every sample in batch
        c_tensor: (B, k_max*n_max*D + k_max + 1)
        k_tensor: (B,) number of non-empty clusters (same scalar for the batch)
    """
    B, D = images.shape
    device = images.device
    centroids_kD, present_mask, clusters_list, k_present = batch_predictor_centroids(images, pred_labels, k_max, D)

    # x: flatten centroids
    x_flat = centroids_kD.reshape(-1)  # (k_max*D,)
    x_tensor = x_flat.unsqueeze(0).repeat(B, 1)  # (B, k_max*D)

    # cluster data padded to n_max per cluster
    cluster_data = torch.zeros(k_max, n_max, D, device=device)
    for i in range(k_max):
        pts = clusters_list[i]
        if pts.shape[0] > 0:
            n_i = min(pts.shape[0], n_max)
            cluster_data[i, :n_i, :] = pts[:n_i]
    cluster_flat = cluster_data.reshape(-1)  # (k_max*n_max*D,)
    cluster_flat_batch = cluster_flat.unsqueeze(0).repeat(B, 1)  # (B, ...)

    # mask + normalized k
    k_mask = present_mask.unsqueeze(0).repeat(B, 1)  # (B, k_max)
    k_real = torch.full((B, 1), fill_value=k_present / k_max, device=device)

    c_tensor = torch.cat([cluster_flat_batch, k_mask, k_real], dim=1)
    k_tensor = torch.full((B,), fill_value=k_present, device=device, dtype=torch.long)
    return x_tensor, c_tensor, k_tensor

# ---------------------------
# Costs / evaluation
# ---------------------------
def k_medians_cost(images, centroids, k_present, k_max=10):
    """
    Sum of L1 distances from each image to the nearest of the first k_present centroids.
    centroids can be (k_max, D) or flattened (k_max*D,).
    """
    B, D = images.shape
    if centroids.dim() == 1:
        centroids = centroids.view(k_max, D)
    valid = centroids[:k_present] if k_present > 0 else centroids[:0]
    if valid.shape[0] == 0:
        return 0.0
    # Manhattan distances (L1)
    dists = torch.cdist(images, valid, p=1)   # (B, k_present)
    return dists.min(dim=1).values.sum().item()


### <<< NEW/CHANGED >>>
@torch.no_grad()
def compute_cost_over_loader(loader, centers_builder, device, k_max=10):
    """
    Generic helper: for each batch, build centroids with `centers_builder(batch_images, batch_labels)`
    and accumulate Euclidean cost to nearest centroid.
    `centers_builder` returns (centroids_kD, k_present)
    """
    total = 0.0
    for Xb, Lb in loader:
        Xb = Xb.to(device)
        Lb = Lb.to(device)
        C_kD, k_present = centers_builder(Xb, Lb)
        total += k_medians_cost(Xb, C_kD, k_present, k_max=k_max)
    return total

### <<< NEW/CHANGED >>>
def centers_from_ground_truth(Xb, yb):
    k_max = 10
    D = Xb.shape[1]
    C, mask, _, k_present = batch_predictor_centroids(Xb, yb, k_max, D)
    return C, k_present

### <<< NEW/CHANGED >>>
def centers_from_predictor(Xb, y_pred_b):
    k_max = 10
    D = Xb.shape[1]
    C, mask, _, k_present = batch_predictor_centroids(Xb, y_pred_b, k_max, D)
    return C, k_present

# ---------------------------
# CVAE (unchanged)
# ---------------------------
class CVAE_Encoder(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc_mu.weight); nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight); nn.init.zeros_(self.fc_logvar.bias)
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
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc_mu.weight); nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight); nn.init.zeros_(self.fc_logvar.bias)
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
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc_out.weight); nn.init.zeros_(self.fc_out.bias)
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
            'mu': mu_post, 'logvar': logvar_post,
            'mu_prior': mu_prior, 'logvar_prior': logvar_prior,
            'z': z,
        }

def cvae_loss(x, output, beta=0.5):
    x_recon = output['x_recon']
    mu = output['mu']; logvar = output['logvar']
    mu_prior = output['mu_prior']; logvar_prior = output['logvar_prior']
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl = -0.5 * torch.sum(
        1 + (logvar - logvar_prior) - ((mu - mu_prior) ** 2 + logvar.exp()) / (logvar_prior.exp() + 1e-10)
    ) / x.size(0)
    return recon_loss + beta * kl, recon_loss, kl

# ---------------------------
# Data (same as yours)
# ---------------------------
set_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
mnist_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

m = 1797
X = torch.stack([mnist_full[i][0] for i in range(m)])  # (1797, 784)
y_true = torch.tensor([mnist_full[i][1] for i in range(m)])  # ground-truth labels

# PCA → 64D
pca = PCA(n_components=64)
X_pca = pca.fit_transform(X.numpy())
X_tensor = torch.tensor(X_pca, dtype=torch.float32)
y_true_tensor = y_true.clone()

# ---------------------------
# Config
# ---------------------------
k_max = 10
n_max = 50
D = 64

input_dim = k_max * D
cond_dim = k_max * n_max * D + k_max + 1
hidden_dim = 512
latent_dim = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# Output & logging
# ---------------------------
save_dir = r'D:\Coding Project\Hephaestus_Algorithms\reconstructions'
os.makedirs(save_dir, exist_ok=True)
print(f"Saving images to: {save_dir}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.normpath(os.path.join(save_dir, f'la_clustering_results_{timestamp}.csv'))
summary_rows = []

# ---------------------------
# Alpha sweep like the paper
# ---------------------------
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # you can change this to match their plots exactly
EPOCHS = 100
BATCH_SIZE = 256

for alpha in alphas:
    print(f"\n=== Running α = {alpha:.1f} ===")
    # Predictor labels
    y_pred = flip_labels(y_true_tensor, alpha=alpha, num_classes=10, seed=1234)

    # Dataloaders:
    #   - train/eval for CVAE use predictor labels (learning-augmented conditioning)
    #   - We also keep a loader with ground-truth labels for OPT when needed
    ds_pred = torch.utils.data.TensorDataset(X_tensor, y_pred)
    ds_true = torch.utils.data.TensorDataset(X_tensor, y_true_tensor)
    train_loader = DataLoader(ds_pred, batch_size=BATCH_SIZE, shuffle=True)
    gt_loader = DataLoader(ds_true, batch_size=BATCH_SIZE, shuffle=False)
    pred_loader = DataLoader(ds_pred, batch_size=BATCH_SIZE, shuffle=False)

    # Baseline costs (computed once per α)
    with torch.no_grad():
        OPT_cost = compute_cost_over_loader(
            gt_loader,
            centers_builder=lambda Xb, Lb: centers_from_ground_truth(Xb, Lb),
            device=device, k_max=k_max
        )
        Predictor_cost = compute_cost_over_loader(
            pred_loader,
            centers_builder=lambda Xb, Lb: centers_from_predictor(Xb, Lb),
            device=device, k_max=k_max
        )
    print(f"[α={alpha:.1f}] OPT (GT centers) cost: {OPT_cost:.4f}")
    print(f"[α={alpha:.1f}] Predictor (noisy) cost: {Predictor_cost:.4f}")

    # Initialize a *fresh* CVAE per α (fair)
    model = CVAE(input_dim=input_dim, cond_dim=cond_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    # Train
    log_rows = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = recon_loss_total = kl_loss_total = 0.0
        count = 0
        for images, pred_labels in train_loader:
            images = images.to(device)
            pred_labels = pred_labels.to(device)

            # Build (x, c) from the *predictor* clusters at the batch level
            x_tensor, c_tensor, k_tensor = prepare_cluster_batch_from_predictor(images, pred_labels, k_max=k_max, n_max=n_max)
            output = model(x_tensor.to(device), c_tensor.to(device))
            loss, recon_loss, kl_loss = cvae_loss(x_tensor.to(device), output, beta=0.5)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
            count += 1

        scheduler.step()

        # Evaluate CVAE cost this epoch: reconstruct batch centroids then cost on images
        model.eval()
        with torch.no_grad():
            cvae_cost_epoch = 0.0
            for images_eval, pred_labels_eval in pred_loader:
                images_eval = images_eval.to(device)
                pred_labels_eval = pred_labels_eval.to(device)
                # Build (x, c) for this eval batch (predictor clusters)
                x_eval, c_eval, k_eval = prepare_cluster_batch_from_predictor(images_eval, pred_labels_eval, k_max=k_max, n_max=n_max)
                out_eval = model(x_eval.to(device), c_eval.to(device))
                # Decode centroids & measure cost to nearest of k_eval centers
                x_recon = out_eval['x_recon'][0]  # any row (same for entire batch)
                k_present = int(k_eval[0].item())
                cvae_cost_epoch += k_medians_cost(images_eval, x_recon, k_present, k_max=k_max)

        row = {
            'Alpha': alpha,
            'Epoch': epoch + 1,
            'Train_Total_Loss': total_loss / max(count, 1),
            'Train_Recon_Loss': recon_loss_total / max(count, 1),
            'Train_KL_Loss': kl_loss_total / max(count, 1),
            'OPT_Cost': OPT_cost,
            'Predictor_Cost': Predictor_cost,
            'CVAE_Cost': cvae_cost_epoch
        }
        log_rows.append(row)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[α={alpha:.1f} | Epoch {epoch+1:03d}] "
                  f"Loss {row['Train_Total_Loss']:.4f} | Recon {row['Train_Recon_Loss']:.4f} | KL {row['Train_KL_Loss']:.4f} | "
                  f"CVAE Cost {row['CVAE_Cost']:.4f} | OPT {OPT_cost:.4f} | Pred {Predictor_cost:.4f}")

    # Save per-α logs and also final summary point (last-epoch CVAE cost)
    df_alpha = pd.DataFrame(log_rows)
    if not os.path.exists(csv_path):
        df_alpha.to_csv(csv_path, index=False)
    else:
        df_alpha.to_csv(csv_path, mode='a', header=False, index=False)

    final_cvae_cost = df_alpha.iloc[-1]['CVAE_Cost']
    summary_rows.append({'Alpha': alpha, 'OPT_Cost': OPT_cost,
                         'Predictor_Cost': Predictor_cost, 'CVAE_Cost': final_cvae_cost})

# Save summary (cost vs α)
summary_df = pd.DataFrame(summary_rows).sort_values('Alpha')
summary_csv = os.path.join(save_dir, f'la_clustering_summary_{timestamp}.csv')
summary_df.to_csv(summary_csv, index=False)
print(f"\nSaved per-epoch log to: {csv_path}")
print(f"Saved summary (cost vs α) to: {summary_csv}")

# Optional: quick plot (cost vs α)
try:
    plt.figure(figsize=(6,4.5))
    plt.plot(summary_df['Alpha'], summary_df['OPT_Cost'], marker='o', label='OPT (Ground Truth)')
    plt.plot(summary_df['Alpha'], summary_df['Predictor_Cost'], marker='o', label='Predictor (Noisy)')
    plt.plot(summary_df['Alpha'], summary_df['CVAE_Cost'], marker='o', label='CVAE (Ours)')
    plt.xlabel('Predictor error rate α')
    plt.ylabel('Clustering cost (Euclidean)')
    plt.title('Cost vs α on MNIST PCA(64), m=1797')
    plt.legend()
    plot_path = os.path.join(save_dir, f'cost_vs_alpha_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()
except Exception as e:
    print("Plotting failed:", e)
