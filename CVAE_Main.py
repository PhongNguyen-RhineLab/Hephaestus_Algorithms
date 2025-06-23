import torch  # Thư viện tính toán tensor
import torch.nn as nn  # Thư viện xây dựng mô hình mạng nơ-ron
import torch.nn.functional as F  # Thư viện các hàm kích hoạt và loss

# --- Encoder q(z|x,c) ---
class CVAE_Encoder(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim):
        super().__init__()  # Khởi tạo lớp cha nn.Module
        self.fc1 = nn.Linear(input_dim + cond_dim, hidden_dim)  # Lớp fully connected đầu vào (gộp x và c)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Lớp tính trung bình (mu) của z
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Lớp tính log phương sai (logvar) của z

    def forward(self, x, c):
        xc = torch.cat([x, c], dim=1)  # Nối x và c theo chiều đặc trưng
        h = F.relu(self.fc1(xc))  # Áp dụng fully connected + hàm kích hoạt ReLU
        mu = self.fc_mu(h)  # Tính trung bình z
        logvar = self.fc_logvar(h)  # Tính log phương sai z
        return mu, logvar  # Trả về hai tham số của phân phối chuẩn

# --- Prior p(z|c) ---
class CVAE_Prior(nn.Module):
    def __init__(self, cond_dim, hidden_dim, latent_dim):
        super().__init__()  # Khởi tạo lớp cha nn.Module
        self.fc1 = nn.Linear(cond_dim, hidden_dim)  # Lớp fully connected cho điều kiện c
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Lớp tính trung bình prior
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Lớp tính log phương sai prior

    def forward(self, c):
        h = F.relu(self.fc1(c))  # Áp dụng fully connected + ReLU cho c
        mu = self.fc_mu(h)  # Trung bình prior
        logvar = self.fc_logvar(h)  # Log phương sai prior
        return mu, logvar  # Trả về hai tham số prior

# --- Decoder p(x|z,c) ---
class CVAE_Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, output_dim):
        super().__init__()  # Khởi tạo lớp cha nn.Module
        self.fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)  # Lớp fully connected đầu vào (z và c)
        self.fc_out = nn.Linear(hidden_dim, output_dim)  # Lớp fully connected ra output

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)  # Nối z và c theo chiều đặc trưng
        h = F.relu(self.fc1(zc))  # Áp dụng fully connected + ReLU
        x_recon = self.fc_out(h)  # Tính output tái tạo
        return x_recon  # Trả về x tái tạo

# --- Energy-Based penalty function (mock) ---
def energy_score(x):
    # Giả lập điểm năng lượng từ mô hình EBM
    return torch.norm(x, dim=1)  # Tính chuẩn L2 của mỗi vector x (có thể thay bằng Eθ(x) thật)

# --- ELBO Loss with Energy Guidance ---
def cvae_loss(x, x_recon, mu, logvar, z_sample, cond, energy_weight=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')  # MSE loss giữa x tái tạo và x gốc
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)  # KL divergence giữa posterior và prior chuẩn
    energy_penalty = energy_score(x_recon).mean()  # Tính điểm năng lượng trung bình
    return recon_loss + kl + energy_weight * energy_penalty  # Tổng loss

class MixCVAE(nn.Module):
    def __init__(self, num_experts, input_dim, cond_dim, hidden_dim, latent_dim):
        super().__init__()  # Khởi tạo lớp cha nn.Module
        self.num_experts = num_experts  # Số lượng chuyên gia (expert)
        self.latent_dim = latent_dim  # Số chiều latent
        self.cond_dim = cond_dim  # Số chiều điều kiện
        self.input_dim = input_dim  # Số chiều đầu vào

        self.encoders = nn.ModuleList([
            CVAE_Encoder(input_dim, cond_dim, hidden_dim, latent_dim) for _ in range(num_experts)
        ])  # Danh sách các encoder cho từng expert
        self.decoders = nn.ModuleList([
            CVAE_Decoder(latent_dim, cond_dim, hidden_dim, input_dim) for _ in range(num_experts)
        ])  # Danh sách các decoder cho từng expert
        self.priors = nn.ModuleList([
            CVAE_Prior(cond_dim, hidden_dim, latent_dim) for _ in range(num_experts)
        ])  # Danh sách các prior cho từng expert

        # Simple uniform gating (can replace by learned gating net)
        self.gate_weights = nn.Parameter(torch.ones(num_experts) / num_experts, requires_grad=False)  # Trọng số chọn expert (đều nhau, không học)

    def sample_expert(self):
        # Sample expert index using uniform or softmax weights
        probs = F.softmax(self.gate_weights, dim=0)  # Tính xác suất chọn expert bằng softmax
        idx = torch.multinomial(probs, 1).item()  # Lấy ngẫu nhiên 1 expert theo phân phối probs
        return idx  # Trả về chỉ số expert

    def forward(self, x, c, expert_idx=None):
        if expert_idx is None:
            expert_idx = self.sample_expert()  # Nếu chưa chỉ định expert thì chọn ngẫu nhiên

        encoder = self.encoders[expert_idx]  # Lấy encoder của expert đã chọn
        decoder = self.decoders[expert_idx]  # Lấy decoder của expert đã chọn
        prior = self.priors[expert_idx]  # Lấy prior của expert đã chọn

        mu, logvar = encoder(x, c)  # Tính tham số phân phối z từ encoder
        std = torch.exp(0.5 * logvar)  # Tính độ lệch chuẩn từ logvar
        z = mu + std * torch.randn_like(std)  # Lấy mẫu z bằng reparameterization trick

        x_recon = decoder(z, c)  # Tái tạo x từ z và c
        mu_prior, logvar_prior = prior(c)  # Tính tham số prior từ c

        return {
            'x_recon': x_recon,  # x tái tạo
            'z': z,  # z đã lấy mẫu
            'mu': mu,  # trung bình posterior
            'logvar': logvar,  # log phương sai posterior
            'mu_prior': mu_prior,  # trung bình prior
            'logvar_prior': logvar_prior,  # log phương sai prior
            'expert_idx': expert_idx  # chỉ số expert đã dùng
        }

def mix_cvae_loss(x, output, energy_weight=1.0, beta=1.0):
    x_recon = output['x_recon']  # x tái tạo
    mu = output['mu']  # trung bình posterior
    logvar = output['logvar']  # log phương sai posterior
    mu_prior = output['mu_prior']  # trung bình prior
    logvar_prior = output['logvar_prior']  # log phương sai prior
    z = output['z']  # z đã lấy mẫu

    recon_loss = F.mse_loss(x_recon, x, reduction='mean')  # MSE loss giữa x tái tạo và x gốc
    # KL divergence giữa posterior và prior có điều kiện
    kl = -0.5 * torch.sum(
        1 + (logvar - logvar_prior) - ((mu - mu_prior) ** 2 + logvar.exp()) / logvar_prior.exp()
    ) / x.size(0)

    energy_penalty = energy_score(x_recon).mean()  # Điểm năng lượng trung bình
    return recon_loss + beta * kl + energy_weight * energy_penalty  # Tổng loss

# Tạo mô hình Mix-CVAE
input_dim = 128  # Số chiều đầu vào x
cond_dim = 64  # Số chiều điều kiện c
hidden_dim = 256  # Số chiều ẩn
latent_dim = 32  # Số chiều latent

model = MixCVAE(num_experts=2, input_dim=input_dim, cond_dim=cond_dim,
                hidden_dim=hidden_dim, latent_dim=latent_dim)  # Khởi tạo mô hình MixCVAE

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Khởi tạo optimizer Adam

# Chạy thử 10 epoch trên dữ liệu ngẫu nhiên
for epoch in range(10):
    x = torch.randn(16, input_dim)  # Sinh batch dữ liệu x ngẫu nhiên
    c = torch.randn(16, cond_dim)  # Sinh batch điều kiện c ngẫu nhiên

    output = model(x, c)  # Forward qua model
    loss = mix_cvae_loss(x, output)  # Tính loss

    optimizer.zero_grad()  # Đặt gradient về 0
    loss.backward()  # Lan truyền gradient ngược
    optimizer.step()  # Cập nhật tham số

    print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f} (Expert {output['expert_idx']})")  # In loss và expert đã chọn
