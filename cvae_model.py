import numpy as np
import random # Ensure random is imported for SimpleMixCVAE

class SimpleCVAE:
    """
    Một triển khai CVAE đơn giản hóa cho mục đích minh họa.
    Đây không phải là một mô hình CVAE thực tế được huấn luyện bằng học sâu.
    """
    def __init__(self, input_dim, latent_dim, condition_dim):
        self.input_dim = input_dim      # Kích thước dữ liệu đầu vào (ví dụ: kích thước của x_perturbation)
        self.latent_dim = latent_dim    # Kích thước không gian tiềm ẩn (z)
        self.condition_dim = condition_dim # Kích thước điều kiện (c - ví dụ: G, K, T được mã hóa)

        # Mô phỏng các trọng số của mạng (sẽ được học trong thực tế)
        # Weights for Encoder: (input_dim + condition_dim) -> latent_dim
        self.encoder_weights_mu = np.random.rand(input_dim + condition_dim, latent_dim)
        self.encoder_weights_logvar = np.random.rand(input_dim + condition_dim, latent_dim)

        # Weights for Decoder: (latent_dim + condition_dim) -> input_dim
        self.decoder_weights = np.random.rand(latent_dim + condition_dim, input_dim)

        # Các giá trị bias (sẽ được học trong thực tế)
        self.encoder_bias_mu = np.random.rand(latent_dim)
        self.encoder_bias_logvar = np.random.rand(latent_dim)
        self.decoder_bias = np.random.rand(input_dim)

    def _flatten_input_x(self, x, c):
        """
        Converts x (dict or array) into a fixed-size numpy array of self.input_dim.
        Requires 'edges' in c for dict-to-vector conversion if x is a dict.
        """
        x_flattened = np.zeros(self.input_dim) # Initialize to zeros

        if isinstance(x, dict):
            if isinstance(c, dict) and 'edges' in c:
                edge_list = c['edges']
                edge_to_idx = {edge: i for i, edge in enumerate(edge_list)}
                for edge, value in x.items():
                    if edge in edge_to_idx:
                        x_flattened[edge_to_idx[edge]] = value
            else:
                # Fallback if 'c' doesn't contain 'edges' for dict conversion
                # This path should ideally be avoided for x_perturbation dicts
                # print("Warning: _flatten_input_x received dict 'x' but 'c' lacks 'edges'. Attempting conversion assuming values map directly to indices.")
                if len(x) == self.input_dim:
                    x_flattened = np.array(list(x.values()))
                else:
                    # Pad/truncate if sizes mismatch
                    temp_x = np.array(list(x.values()))
                    x_flattened[:min(len(temp_x), self.input_dim)] = temp_x[:min(len(temp_x), self.input_dim)]
        else:
            # Assume x is already a numpy array or list
            x_array = np.asarray(x)
            if x_array.shape[0] != self.input_dim:
                # Pad/truncate if sizes mismatch
                x_flattened[:min(x_array.shape[0], self.input_dim)] = x_array[:min(x_array.shape[0], self.input_dim)]
            else:
                x_flattened = x_array
        return x_flattened

    def _vectorize_condition(self, c):
        """
        Converts the condition 'c' (dict or array) into a fixed-size numpy array of self.condition_dim.
        """
        if isinstance(c, dict):
            # As in the previous _combine_input_and_condition, this is a mock.
            # In a real CVAE, 'c' (graph, K, T) would be encoded via a GNN or MLP
            # into a fixed-size vector of 'condition_dim'.
            # For this mock, we ensure it returns a vector of self.condition_dim
            return np.random.rand(self.condition_dim) # Placeholder for complex condition encoding
        else:
            # Assume it's already a vector and check shape if necessary
            c_vec = np.asarray(c)
            if c_vec.shape[0] != self.condition_dim:
                # Handle size mismatch if 'c' is passed as a direct vector
                if c_vec.shape[0] < self.condition_dim:
                    temp_c = np.zeros(self.condition_dim)
                    temp_c[:c_vec.shape[0]] = c_vec
                    c_vec = temp_c
                else:
                    c_vec = c_vec[:self.condition_dim]
            return c_vec

    def encode(self, x, c):
        """
        Mã hóa dữ liệu đầu vào x với điều kiện c thành các tham số của phân phối tiềm ẩn (mu, logvar).
        Trong thực tế, đây là một mạng nơ-ron nhiều lớp.
        """
        x_flat = self._flatten_input_x(x, c)
        c_vec = self._vectorize_condition(c)

        combined_input = np.concatenate((x_flat, c_vec)) # Input for encoder weights

        mu = np.dot(combined_input, self.encoder_weights_mu) + self.encoder_bias_mu
        logvar = np.dot(combined_input, self.encoder_weights_logvar) + self.encoder_bias_logvar

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Kỹ thuật reparameterization trick để lấy mẫu từ phân phối tiềm ẩn.
        z = mu + exp(0.5 * logvar) * epsilon, với epsilon ~ N(0, 1)
        """
        std = np.exp(0.5 * logvar)
        epsilon = np.random.randn(*std.shape) # Lấy mẫu từ phân phối chuẩn
        z = mu + std * epsilon
        return z

    def decode(self, z, c):
        """
        Giải mã latent vector z với điều kiện c thành dữ liệu đầu ra tái tạo.
        Trong thực tế, đây là một mạng nơ-ron nhiều lớp.
        """
        vectorized_c = self._vectorize_condition(c)
        combined_latent_and_condition = np.concatenate((z, vectorized_c)) # Input for decoder weights

        reconstructed_x_raw_vec = np.dot(combined_latent_and_condition, self.decoder_weights) + self.decoder_bias

        # Now convert reconstructed_x_raw_vec back to the expected dict format {edge_id: value}
        reconstructed_x_dict = {}
        if isinstance(c, dict) and 'edges' in c:
            for i, edge_id in enumerate(c['edges']):
                if i < len(reconstructed_x_raw_vec): # Ensure not out of bounds
                    reconstructed_x_dict[edge_id] = max(0, int(np.round(reconstructed_x_raw_vec[i]))) # Non-negative integer
        else:
            # Fallback if 'c' doesn't contain 'edges', return a generic dict
            reconstructed_x_dict = {f"edge_{i}": max(0, int(np.round(val))) for i, val in enumerate(reconstructed_x_raw_vec)}

        return reconstructed_x_dict

    def forward(self, x, c):
        """Thực hiện quá trình forward của CVAE."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z, c)
        return reconstructed_x, mu, logvar

    def loss_function(self, reconstructed_x_raw, x, mu, logvar, c): # Added 'c' as arg
        """
        Tính toán hàm mất mát ELBO (Evidence Lower Bound).
        ELBO = Reconstruction_Loss + KL_Divergence (for VAEs, minimize negative ELBO)
        """
        # Ensure x is flattened for reconstruction loss comparison
        x_true_vec = self._flatten_input_x(x, c)

        # Ensure reconstructed_x_raw is flattened if it's a dict
        recon_x_vec = np.zeros(self.input_dim)
        if isinstance(reconstructed_x_raw, dict):
            recon_x_vec = self._flatten_input_x(reconstructed_x_raw, c)
        else:
            recon_x_vec = np.asarray(reconstructed_x_raw)
            # Ensure recon_x_vec shape matches self.input_dim for MSE
            if recon_x_vec.shape[0] != self.input_dim:
                 temp_recon = np.zeros(self.input_dim)
                 temp_recon[:min(recon_x_vec.shape[0], self.input_dim)] = recon_x_vec[:min(recon_x_vec.shape[0], self.input_dim)]
                 recon_x_vec = temp_recon

        reconstruction_loss = np.mean(np.square(x_true_vec - recon_x_vec))

        # KL Divergence (KL(q(z|x,c) || p(z|c)))
        # Assuming p(z|c) is a standard normal distribution N(0, I)
        kl_divergence = -0.5 * np.sum(1 + logvar - np.square(mu) - np.exp(logvar))

        elbo = reconstruction_loss + kl_divergence
        return elbo, reconstruction_loss, kl_divergence

# --- Lớp để mô phỏng một hỗn hợp các CVAE (Mix-CVAE) ---
class SimpleMixCVAE:
    def __init__(self, input_dim, latent_dim, condition_dim, num_experts=1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.experts = []
        for _ in range(num_experts):
            self.experts.append(SimpleCVAE(input_dim, latent_dim, condition_dim))

    def _get_active_expert(self, c):
        """
        Trong thực tế, một mạng gating sẽ chọn expert dựa trên điều kiện c.
        Ở đây, chúng ta sẽ chọn một expert ngẫu nhiên hoặc expert đầu tiên.
        """
        return random.choice(self.experts)

    def sample_latent(self, c):
        """Lấy mẫu latent vector từ một expert ngẫu nhiên (hoặc expert được chọn)."""
        # expert = self._get_active_expert(c) # Not needed for sampling from prior
        # This is p_phi(z|c) from the Morph algorithm. Assumed to be N(0,I)
        return np.random.randn(self.latent_dim)

    def encode(self, x, c):
        """Mã hóa x và c sử dụng expert hoạt động."""
        expert = self._get_active_expert(c)
        return expert.encode(x, c)

    def decode(self, z, c):
        """Giải mã z và c sử dụng expert hoạt động."""
        expert = self._get_active_expert(c)
        return expert.decode(z, c)

    def add_expert(self):
        """Thêm một expert CVAE mới vào hỗn hợp."""
        self.experts.append(SimpleCVAE(self.input_dim, self.latent_dim, self.condition_dim))
        print(f"Added new SimpleCVAE expert. Total experts: {len(self.experts)}")