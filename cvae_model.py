import numpy as np

import random

class SimpleCVAE:
    """
    Một triển khai CVAE đơn giản hóa cho mục đích minh họa.
    Đây không phải là một mô hình CVAE thực tế được huấn luyện bằng học sâu.
    """

    def __init__(self, input_dim, latent_dim, condition_dim):
        self.input_dim = input_dim  # Kích thước dữ liệu đầu vào (ví dụ: kích thước của x_perturbation)
        self.latent_dim = latent_dim  # Kích thước không gian tiềm ẩn (z)
        self.condition_dim = condition_dim  # Kích thước điều kiện (c - ví dụ: G, K, T được mã hóa)

        # Mô phỏng các trọng số của mạng (sẽ được học trong thực tế)
        # Chúng ta sẽ sử dụng các ma trận ngẫu nhiên đơn giản
        self.encoder_weights_mu = np.random.rand(input_dim + condition_dim, latent_dim)
        self.encoder_weights_logvar = np.random.rand(input_dim + condition_dim, latent_dim)
        self.decoder_weights = np.random.rand(latent_dim + condition_dim, input_dim)

        # Các giá trị bias (sẽ được học trong thực tế)
        self.encoder_bias_mu = np.random.rand(latent_dim)
        self.encoder_bias_logvar = np.random.rand(latent_dim)
        self.decoder_bias = np.random.rand(input_dim)

    def _combine_input_and_condition(self, x, c):
        """Kết hợp input (x) và điều kiện (c) cho encoder/decoder."""
        # Chuyển dictionary x thành array (giả định thứ tự cố định)
        # Trong thực tế, cần một cách mã hóa x_perturbation thành vector
        # Ở đây, giả định x là một dict và chúng ta sẽ lấy sum hoặc một số giá trị đại diện
        if isinstance(x, dict):
            # Cực kỳ đơn giản hóa: chuyển x_perturbation dict thành một vector 1 chiều
            # Để demo, chúng ta sẽ làm phẳng hoặc lấy một số thuộc tính
            # Giả sử input_dim là tổng số cạnh, và x là dict {edge_id: value}
            # Cần một ánh xạ cố định từ edge_id đến index
            x_vec = np.zeros(self.input_dim)
            # Đây là phần mô phỏng nhất: cách map dict x_perturbation -> vector input
            # Cho ví dụ này, chúng ta sẽ giả định x là một vector numpy hoặc list
            # Để đơn giản hóa hơn, hãy giả sử x là một list/array có kích thước input_dim
            # Nếu x là dict, bạn phải có một cách chuyển nó thành vector số
            # Ví dụ: x_vec = np.array(list(x.values()))[:self.input_dim]
            # Hoặc, một cách đáng tin cậy hơn là hàm gọi sẽ chuyển đổi
            x_flattened = x  # Sẽ giả định x đã là numpy array hoặc list

        # Chuyển điều kiện c thành một vector
        # Ở đây, giả định c là một dict, chúng ta sẽ làm phẳng nó
        # Trong thực tế, c (graph, K, T) sẽ được mã hóa phức tạp (ví dụ: GNN, MLP)
        if isinstance(c, dict):
            # Ví dụ cực kỳ đơn giản: lấy tổng của T_rand và số lượng cặp K_rand
            # Hoặc sử dụng một vector ngẫu nhiên
            c_vec = np.random.rand(self.condition_dim)  # Mô phỏng
        else:
            c_vec = c  # Giả định c đã là vector

        return np.concatenate((x_flattened, c_vec))

    def encode(self, x, c):
        """
        Mã hóa dữ liệu đầu vào x với điều kiện c thành các tham số của phân phối tiềm ẩn (mu, logvar).
        Trong thực tế, đây là một mạng nơ-ron nhiều lớp.
        """
        combined_input = self._combine_input_and_condition(x, c)

        # Mô phỏng phép nhân ma trận và cộng bias
        mu = np.dot(combined_input, self.encoder_weights_mu) + self.encoder_bias_mu
        logvar = np.dot(combined_input, self.encoder_weights_logvar) + self.encoder_bias_logvar

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Kỹ thuật reparameterization trick để lấy mẫu từ phân phối tiềm ẩn.
        z = mu + exp(0.5 * logvar) * epsilon, với epsilon ~ N(0, 1)
        """
        std = np.exp(0.5 * logvar)
        epsilon = np.random.randn(*std.shape)  # Lấy mẫu từ phân phối chuẩn
        z = mu + std * epsilon
        return z

    def decode(self, z, c):
        """
        Giải mã latent vector z với điều kiện c thành dữ liệu đầu ra tái tạo.
        Trong thực tế, đây là một mạng nơ-ron nhiều lớp.
        """
        combined_latent_and_condition = np.concatenate(
            (z, self._combine_input_and_condition(np.zeros(self.input_dim), c)))

        # Mô phỏng phép nhân ma trận và cộng bias
        reconstructed_x = np.dot(combined_latent_and_condition, self.decoder_weights) + self.decoder_bias

        # Đảm bảo reconstructed_x có dạng dictionary nếu đầu vào mong đợi là dict
        # Đây là phần rất tùy chỉnh theo cách bạn muốn output.
        # Đối với x_perturbation, nó phải là non-negative integers
        reconstructed_x_dict = {}
        # Cực kỳ đơn giản: giả định output_dim là số lượng cạnh của đồ thị test
        # Và chúng ta cần ánh xạ các giá trị này trở lại các edge_id
        # Để đơn giản cho demo, ta sẽ trả về một dict ngẫu nhiên (hoặc làm tròn các giá trị)
        # Giả định c chứa 'edges'
        if isinstance(c, dict) and 'edges' in c:
            for i, edge_id in enumerate(c['edges']):
                if i < len(reconstructed_x):  # Đảm bảo không vượt quá kích thước
                    reconstructed_x_dict[edge_id] = max(0, int(np.round(
                        reconstructed_x[i])))  # Đảm bảo non-negative integer
        else:
            # Nếu không có thông tin cạnh, trả về một dict mô phỏng
            reconstructed_x_dict = {f"edge_{i}": max(0, int(np.round(val))) for i, val in enumerate(reconstructed_x)}

        return reconstructed_x_dict

    def forward(self, x, c):
        """Thực hiện quá trình forward của CVAE."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z, c)
        return reconstructed_x, mu, logvar

    def loss_function(self, reconstructed_x_raw, x, mu, logvar):
        """
        Tính toán hàm mất mát ELBO (Evidence Lower Bound).
        ELBO = Reconstruction_Loss - KL_Divergence
        """
        # Reconstruction Loss (ví dụ: MSE hoặc Binary Cross-Entropy)
        # Cần chuyển reconstructed_x_raw và x về dạng vector tương ứng để so sánh

        # Demo đơn giản: chuyển dicts thành list/array
        # Đây là phần quan trọng nhất để CVAE thực sự học được.
        # Cần có ánh xạ cố định từ edge_id đến index.
        # Giả sử x_true_vec và recon_x_vec đã được chuẩn bị đúng kích thước input_dim

        x_true_vec = np.zeros(self.input_dim)
        if isinstance(x, dict):
            # Lấy các giá trị từ x (dict) và đưa vào vector
            # Cần một ánh xạ cố định cho các cạnh của đồ thị.
            # Ví dụ: x_true_vec = np.array(list(x.values()))[:self.input_dim]
            # Giả định x là một list/array cho đơn giản
            x_true_vec = x  # Giả định x đã là numpy array hoặc list

        recon_x_vec = np.zeros(self.input_dim)
        if isinstance(reconstructed_x_raw, dict):
            # Lấy các giá trị từ reconstructed_x_raw (dict) và đưa vào vector
            # Giả định reconstructed_x_raw là một list/array
            recon_x_vec = np.array(list(reconstructed_x_raw.values()))[:self.input_dim]
        else:
            recon_x_vec = reconstructed_x_raw  # Giả định đã là numpy array

        # Mean Squared Error (MSE) cho Reconstruction Loss
        reconstruction_loss = np.mean(np.square(x_true_vec - recon_x_vec))

        # KL Divergence (KL(q(z|x,c) || p(z|c)))
        # Giả định p(z|c) là phân phối chuẩn N(0, I)
        kl_divergence = -0.5 * np.sum(1 + logvar - np.square(mu) - np.exp(logvar))

        elbo = reconstruction_loss + kl_divergence  # ELBO là tổng của 2 thành phần này
        return elbo, reconstruction_loss, kl_divergence


# --- Lớp để mô phỏng một hỗn hợp các CVAE (Mix-CVAE) ---
# Tương tự như MockMixCVAE đã có, nhưng giờ dùng SimpleCVAE

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
        expert = self._get_active_expert(c)
        # Để lấy mẫu từ phân phối tiềm ẩn của CVAE, cần encode một "x_mẫu"
        # Tuy nhiên, nếu chỉ cần lấy mẫu z từ prior, chúng ta lấy mẫu trực tiếp từ N(0,I)
        # Đây là p_phi(z|c) từ thuật toán Morph
        return np.random.randn(self.latent_dim)  # Lấy mẫu trực tiếp từ N(0,I) prior

    def encode(self, x, c):
        """Mã hóa x và c sử dụng expert hoạt động."""
        expert = self._get_active_expert(c)  # Lựa chọn expert dựa trên c
        return expert.encode(x, c)

    def decode(self, z, c):
        """Giải mã z và c sử dụng expert hoạt động."""
        expert = self._get_active_expert(c)  # Lựa chọn expert dựa trên c
        return expert.decode(z, c)

    def add_expert(self):
        """Thêm một expert CVAE mới vào hỗn hợp."""
        self.experts.append(SimpleCVAE(self.input_dim, self.latent_dim, self.condition_dim))
        print(f"Added new SimpleCVAE expert. Total experts: {len(self.experts)}")