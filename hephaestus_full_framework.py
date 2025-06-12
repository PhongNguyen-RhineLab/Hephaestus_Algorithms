import networkx as nx
import numpy as np
from cvae_model import SimpleCVAE
from collections import defaultdict
import random


# --- 0. Các hàm hỗ trợ chung ---

def dijkstra_path_cost(G, s, t, x_perturbation, edge_cost_functions):
    """
    Tính toán chi phí đường đi ngắn nhất từ s đến t trên đồ thị G,
    sử dụng trọng số cạnh được điều chỉnh bởi x_perturbation và edge_cost_functions.
    Trả về chi phí và đường đi.
    """
    G_temp = nx.DiGraph()
    G_temp.add_nodes_from(G.nodes)
    for u, v, data in G.edges(data=True):
        edge_id = (u, v)  # Giả sử edge_id là một tuple (u, v)
        # Chi phí của cạnh được xác định bởi hàm chi phí cạnh và giá trị nhiễu x_perturbation
        # Nếu x_perturbation không chứa edge_id, giả sử giá trị nhiễu là 0
        perturb_value = x_perturbation.get(edge_id, 0)
        # Sử dụng hàm chi phí được cung cấp cho cạnh đó, hoặc một hàm mặc định nếu không có.
        cost_func = edge_cost_functions.get(edge_id, lambda x: x)  # Mặc định là tuyến tính
        cost = cost_func(perturb_value + data.get('base_weight', 0))  # Cộng nhiễu vào trọng số cơ bản
        G_temp.add_edge(u, v, weight=cost)

    try:
        path = nx.dijkstra_path(G_temp, source=s, target=t, weight='weight')
        cost = nx.dijkstra_path_length(G_temp, source=s, target=t, weight='weight')
        return cost, path
    except nx.NetworkXNoPath:
        return float('inf'), []  # Trả về vô cùng nếu không có đường đi


def get_violating_pairs(G, K, T, x_perturbation, edge_cost_functions, spagan_model=None, use_spagan=False):
    """
    Xác định các cặp (s,t) trong K mà chi phí đường đi ngắn nhất của chúng
    thấp hơn ngưỡng T.
    Nếu use_spagan là True, sử dụng spagan_model để ước tính chi phí.
    Nếu không, sử dụng dijkstra_path_cost.
    """
    violating_pairs = set()
    for s, t in K:
        if use_spagan:
            # Mô phỏng SPAGAN: Trả về chi phí ước tính
            estimated_cost = spagan_model(G, s, t, x_perturbation)  #
            if estimated_cost < T:
                violating_pairs.add((s, t))
        else:
            cost, _ = dijkstra_path_cost(G, s, t, x_perturbation, edge_cost_functions)
            if cost < T:
                violating_pairs.add((s, t))
    return violating_pairs


def compute_soft_potential_function(P_paths, x_perturbation, T, spagan_model=None, G=None, edge_cost_functions=None,
                                    use_spagan=False):
    """
    Tính toán hàm tiềm năng mềm C(P,x).
    P_paths là tập hợp các đường đi ngắn nhất đã tìm thấy cho các cặp vi phạm.
    """
    C_Px = 0
    for path_s_t in P_paths:
        s, t = path_s_t[0], path_s_t[-1]  # Lấy điểm nguồn và đích từ đường đi
        if use_spagan:
            # spagan_model cần ước tính chi phí cho một cặp (s,t) và x
            cost_st = spagan_model(G, s, t, x_perturbation)
        else:
            cost_st, _ = dijkstra_path_cost(G, s, t, x_perturbation, edge_cost_functions)
        C_Px += min(T, cost_st)
    return C_Px


def find_optimal_increase(G, P_paths, x_perturbation, T, b_budgets, soft_epsilon, spagan_model=None,
                          edge_cost_functions=None, use_spagan=False):
    """
    Tìm cạnh e* và mức tăng Delta* dẫn đến mức tăng tiềm năng lớn nhất trên một đơn vị ngân sách.
    """
    E_P = set()  # Tập hợp các cạnh trong tất cả các đường đi trong P
    for path_s_t in P_paths:
        for i in range(len(path_s_t) - 1):
            E_P.add((path_s_t[i], path_s_t[i + 1]))

    e_star = None
    delta_star = 0
    max_delta_ratio = -float('inf')

    C_Px_current = compute_soft_potential_function(P_paths, x_perturbation, T, spagan_model, G, edge_cost_functions,
                                                   use_spagan)

    for e in E_P:
        u, v = e
        max_delta_e = b_budgets.get(e, 0) - x_perturbation.get(e, 0)
        print(f"      find_optimal_increase: Edge {e}, current x={x_perturbation.get(e, 0)}, max_delta_e={max_delta_e}")
        for delta in range(1, int(max_delta_e) + 1):  # Delta từ 1 đến max_delta_e
            x_prime = x_perturbation.copy()
            x_prime[e] = x_prime.get(e, 0) + delta

            print(f"        find_optimal_increase: Trying delta={delta} for edge {e}, x_prime[{e}]={x_prime[e]}")
            C_Px_prime = compute_soft_potential_function(P_paths, x_prime, T, spagan_model, G, edge_cost_functions,
                                                         use_spagan)
            print(f"        find_optimal_increase: C_Px_prime={C_Px_prime:.4f}")

            if delta > 0:  # Tránh chia cho 0
                delta_ratio = (C_Px_prime - C_Px_current) / delta
                if delta_ratio > max_delta_ratio:
                    max_delta_ratio = delta_ratio
                    e_star = e
                    delta_star = delta

    return e_star, delta_star, max_delta_ratio


# --- Hàm chi phí cạnh (f_e) ---
# Tương ứng với phần 3.7 trong tài liệu bổ sung  và phần "Edge Weight Function Settings" trong bài báo chính.

def linear_edge_cost(x_e):
    """Hàm chi phí cạnh tuyến tính: f_e(x_e) = x_e"""  #
    return x_e


def quadratic_edge_cost(x_e):
    """Hàm chi phí cạnh bậc hai: f_e(x_e) = x_e^2"""  #
    return x_e ** 2


def log_concave_edge_cost(x_e):
    """Hàm chi phí cạnh log-concave: f_e(x_e) = log(1 + x_e)"""  #
    return np.log(1 + x_e) if x_e >= 0 else -np.inf


# --- 1. Thuật toán 3: SPAGAN Training (Mô phỏng) ---
# Trong thực tế, SPAGAN (Shortest Path Graph Attention Network) là một GNN phức tạp.
# Để đơn giản, chúng ta mô phỏng nó bằng một hàm trả về ước tính chi phí.

def mock_spagan_model(G, s, t, x_perturbation):
    """
    Mô phỏng mô hình SPAGAN (F_theta).
    Ước tính chi phí đường đi giữa s và t.
    Để làm cho C(P,x) tăng lên, chúng ta sẽ làm một mô phỏng đơn giản hơn:
    Nếu có đường đi trực tiếp (s,t), chi phí sẽ được ước tính trên đường đi đó.
    Nếu không, sẽ dùng Dijkstra.
    """
    print(f"  SPAGAN: Estimating path from {s} to {t} with x_perturbation={x_perturbation}")

    edge_id = (s, t)
    if G.has_edge(s, t):
        base_weight = G.edges[s, t].get('base_weight', 1.0)
        perturb_value = x_perturbation.get(edge_id, 0)
        estimated_cost = base_weight + perturb_value * 2.0
        print(
            f"    SPAGAN_Direct_Edge: {edge_id}, base={base_weight}, perturb={perturb_value}, estimated_cost={estimated_cost}")
        print(f"  SPAGAN: Estimated path cost {s}->{t} (direct) = {estimated_cost}")
        return estimated_cost
    else:
        # Nếu không có đường đi trực tiếp, chúng ta vẫn cần tìm đường đi ngắn nhất
        # với các trọng số được điều chỉnh bởi nhiễu.
        G_temp = nx.DiGraph()
        G_temp.add_nodes_from(G.nodes)
        for u, v, data in G.edges(data=True):
            edge_id = (u, v)
            base_weight = data.get('base_weight', 1.0)
            perturb_value = x_perturbation.get(edge_id, 0)
            estimated_cost_on_edge = base_weight + perturb_value * 2.0
            G_temp.add_edge(u, v, weight=estimated_cost_on_edge)

        try:
            cost = nx.dijkstra_path_length(G_temp, s, t, weight='weight')
            print(f"  SPAGAN: Estimated path cost {s}->{t} (Dijkstra) = {cost}")
            return cost
        except nx.NetworkXNoPath:
            print(f"  SPAGAN: No path found {s}->{t}. Returning inf.")
            return float('inf')


# --- 2. Thuật toán 4: Predictive Path Stressing (PPS) ---
# Được sử dụng trong giai đoạn "Forge".

def predictive_path_stressing(G, K, T, edge_cost_functions, b_budgets, x_initial, F_theta, soft_epsilon=1e-6):
    """
    Thực hiện thuật toán Predictive Path Stressing (PPS).
    G: Đồ thị NetworkX.
    K: Tập hợp các cặp (s, t) quan trọng.
    T: Ngưỡng chi phí.
    edge_cost_functions: Dictionary các hàm chi phí cạnh, key là edge_id (u,v).
    b_budgets: Dictionary các giới hạn ngân sách tối đa cho mỗi cạnh, key là edge_id (u,v).
    x_initial: Vector nhiễu ban đầu (dictionary edge_id -> giá trị nhiễu).
    F_theta: Mô hình SPAGAN đã huấn luyện (hàm).
    soft_epsilon: Tham số cho ngưỡng mềm của hàm tiềm năng.
    """
    x = x_initial.copy()

    # Bước 2 & 3: Xác định các cặp vi phạm.
    # Sử dụng SPAGAN để ước tính chi phí.
    K_violate = get_violating_pairs(G, K, T, x, edge_cost_functions, spagan_model=F_theta, use_spagan=True)

    while K_violate:
        print(f"    PPS: Còn {len(K_violate)} cặp vi phạm. Đang tối ưu hóa tiềm năng...")  # Hiện tiến độ
        P_paths = []  # Tập hợp các đường đi ngắn nhất cho các cặp vi phạm.
        for s, t in K_violate:
            # Trong PPS, SPAGAN được dùng để tìm đường đi và ước tính chi phí.
            # Để đơn giản hóa mô phỏng, chúng ta sẽ sử dụng Dijkstra với trọng số từ SPAGAN
            # để có đường đi.
            G_spagan_weighted = nx.DiGraph()
            G_spagan_weighted.add_nodes_from(G.nodes)
            for u, v, data in G.edges(data=True):
                edge_id = (u, v)
                # SPAGAN ước tính chi phí cho cạnh (u,v)
                estimated_cost_on_edge = F_theta(G, u, v, x)
                G_spagan_weighted.add_edge(u, v, weight=estimated_cost_on_edge)

            try:
                rho_s_t = nx.dijkstra_path(G_spagan_weighted, s, t, weight='weight')
            except nx.NetworkXNoPath:
                rho_s_t = []  # Không có đường đi

            if rho_s_t:
                P_paths.append(rho_s_t)

        # Bước 9-24: Đánh giá hàm tiềm năng mềm và cập nhật x.
        current_C_Px = compute_soft_potential_function(P_paths, x, T, spagan_model=F_theta, G=G, use_spagan=True)

        # Lặp lại cho đến khi C(P,x) đạt đến ngưỡng mềm.
        while current_C_Px < len(P_paths) * T - soft_epsilon:
            print(
                f"      PPS: C(P,x) = {current_C_Px:.4f}, Ngưỡng = {len(P_paths) * T - soft_epsilon:.4f}. Tìm mức tăng tối ưu...")  # Hiện tiến độ
            e_star, delta_star, _ = find_optimal_increase(G, P_paths, x, T, b_budgets, soft_epsilon,
                                                          spagan_model=F_theta, use_spagan=True)

            if e_star is None or delta_star == 0:  # Không tìm thấy cải thiện nào
                break

            x[e_star] = x.get(e_star, 0) + delta_star
            current_C_Px = compute_soft_potential_function(P_paths, x, T, spagan_model=F_theta, G=G, use_spagan=True)

        # Bước 25: Cập nhật các cặp vi phạm.
        K_violate = get_violating_pairs(G, K, T, x, edge_cost_functions, spagan_model=F_theta, use_spagan=True)

    return x


# --- 3. Thuật toán 5: Morph (Mô phỏng) ---
# Giai đoạn này huấn luyện EBM và Mix-CVAE.

class MockEBM:
    """Mô phỏng Energy-Based Model (EBM)."""

    def __init__(self):
        # EBM học một hàm năng lượng E(x)
        pass

    def energy(self, x):
        # Mô phỏng năng lượng của x. Giá trị năng lượng thấp cho giải pháp tốt.
        # Để đơn giản, năng lượng phụ thuộc vào tổng giá trị nhiễu.
        return sum(x.values()) * 0.1  # Năng lượng càng thấp khi x càng nhỏ


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
        return np.random.randn(self.latent_dim) # Lấy mẫu trực tiếp từ N(0,I) prior

    def encode(self, x, c):
        """Mã hóa x và c sử dụng expert hoạt động."""
        expert = self._get_active_expert(c) # Lựa chọn expert dựa trên c
        return expert.encode(x, c)

    def decode(self, z, c):
        """Giải mã z và c sử dụng expert hoạt động."""
        expert = self._get_active_expert(c) # Lựa chọn expert dựa trên c
        return expert.decode(z, c)

    def add_expert(self):
        """Thêm một expert CVAE mới vào hỗn hợp."""
        self.experts.append(SimpleCVAE(self.input_dim, self.latent_dim, self.condition_dim))
        print(f"Added new SimpleCVAE expert. Total experts: {len(self.experts)}")


def morph_phase(D_sol, K_max_morph_iter=100, N_initial_experts=1):
    print("\n--- Bắt đầu giai đoạn Morph (Mô phỏng) ---")
    q_ebm = MockEBM()  # Mô phỏng EBM

    # Định nghĩa kích thước cho CVAE:
    # input_dim: Kích thước của vector x_perturbation. Đây sẽ là số lượng cạnh trong đồ thị lớn nhất
    # hoặc một kích thước cố định mà bạn mã hóa x_perturbation vào.
    # Để đơn giản, hãy lấy số cạnh của đồ thị đầu tiên trong D_graph_data.
    # Trong thực tế, bạn cần một kích thước cố định cho tất cả các đồ thị hoặc một cơ chế xử lý đồ thị khác nhau.

    # Để đơn giản cho demo, lấy input_dim từ một đồ thị mẫu hoặc định nghĩa cố định
    # Giả sử chúng ta có một đồ thị mẫu để lấy thông tin về số cạnh
    sample_graph = D_sol[0]['graph'] if D_sol else None
    if sample_graph:
        input_dim_cvae = len(sample_graph.edges())  # Số lượng cạnh
    else:
        input_dim_cvae = 10  # Giá trị mặc định nếu D_sol rỗng

    latent_dim_cvae = 128  # Kích thước không gian tiềm ẩn (thường được đặt là 128 hoặc 256)
    condition_dim_cvae = 5  # Kích thước vector điều kiện (ví dụ: mã hóa G, K, T)

    # Sử dụng SimpleMixCVAE thay vì MockMixCVAE
    omega_mix_cvae = SimpleMixCVAE(input_dim_cvae, latent_dim_cvae, condition_dim_cvae, num_experts=N_initial_experts)

    for k in range(K_max_morph_iter):
        if not D_sol:
            print("D_sol is empty, skipping Morph phase updates.")
            break

        selected_sol = random.choice(D_sol)
        G_rand = selected_sol['graph']
        K_rand = selected_sol['K']
        T_rand = selected_sol['T']
        x_real_rand_dict = selected_sol['x_sol']
        edge_cost_functions_rand = selected_sol['edge_cost_functions']

        # Chuẩn bị input x và điều kiện c cho CVAE
        # CHUYỂN ĐỔI x_real_rand_dict THÀNH VECTOR (rất quan trọng)
        # Đây là phần cần ánh xạ cố định từ edge_id đến index.
        # Để demo, chúng ta sẽ tạo một vector từ dict bằng cách lấy tất cả các giá trị của edge_id
        # Nếu số lượng cạnh khác nhau giữa các đồ thị, bạn cần một cách nhất quán để mã hóa.
        # Ở đây, giả định input_dim_cvae là số cạnh trong đồ thị hiện tại G_rand

        # Để đơn giản hóa, ta sẽ tạo một vector từ các cạnh trong G_rand và x_real_rand_dict
        x_real_rand_vec = np.zeros(input_dim_cvae)
        # Giả định một ánh xạ tạm thời từ edge_id tới index (rất đơn giản cho demo)
        edge_to_idx = {edge: i for i, edge in enumerate(G_rand.edges())}
        for edge, value in x_real_rand_dict.items():
            if edge in edge_to_idx:
                x_real_rand_vec[edge_to_idx[edge]] = value

        # Chuẩn bị điều kiện c_context
        # Cực kỳ đơn giản hóa: chuyển điều kiện thành một vector số
        c_context = {
            'graph': G_rand,
            'K': K_rand,
            'T': T_rand,
            'edges': list(G_rand.edges()),  # Cần để decode biết các cạnh
            'edge_cost_functions': edge_cost_functions_rand  # Để tinh chỉnh sau này
        }
        # Mã hóa c thành vector (mô phỏng)
        c_vec = np.random.rand(condition_dim_cvae)  # Điều kiện phức tạp sẽ được mã hóa

        # --- Huấn luyện EBM và Mix-CVAE ---
        # 1. Update EBM (q_ebm)
        # Sample fake data (x_fake) from omega_mix_cvae
        z_fake = omega_mix_cvae.sample_latent(c_context)  # Lấy mẫu latent từ prior
        x_fake_rand_dict = omega_mix_cvae.decode(z_fake, c_context)  # Decode thành dict

        # Giả lập tối ưu hóa EBM:
        # Trong thực tế, L_q = E_real[E(x_real)] - E_fake[E(x_fake)] + regularization
        real_energy = q_ebm.energy(x_real_rand_dict)
        fake_energy = q_ebm.energy(x_fake_rand_dict)
        # (Ở đây, EBM sẽ điều chỉnh trọng số của nó để real_energy thấp và fake_energy cao)

        # 2. Update Mix-CVAE (omega_mix_cvae)
        # Tính ELBO cho x_real_rand_vec và c_vec
        # (Ở đây, omega_mix_cvae sẽ điều chỉnh trọng số của nó để ELBO tối ưu)
        reconstructed_x_real, mu, logvar = omega_mix_cvae.forward(x_real_rand_vec, c_vec)
        elbo_loss, recon_loss, kl_loss = omega_mix_cvae.loss_function(reconstructed_x_real, x_real_rand_vec, mu, logvar)

        # Thêm energy penalty: L_Omega_guide = L_Omega_ELBO + lambda * E_generated[E_ebm(x)]
        # x_generated = omega_mix_cvae.decode(z_generated_from_prior, c)
        # generated_energy = q_ebm.energy(x_generated)
        # total_cvae_loss = elbo_loss + lambda_factor * generated_energy # Ví dụ về cách kết hợp

        # Mô phỏng chiến lược thêm expert (Expert Addition Strategy)
        if k % 20 == 0 and k > 0:
            omega_mix_cvae.add_expert()

    print("--- Kết thúc giai đoạn Morph ---")
    return q_ebm, omega_mix_cvae


# --- 4. Thuật toán 8: Predictive Path Stressing - Inference (PPS-I) ---
# Được sử dụng trong giai đoạn "Refine" và "Inference".

def predictive_path_stressing_inference(G, K, T, edge_cost_functions, b_budgets, x_initial, soft_epsilon=1e-6):
    """
    Thực hiện thuật toán Predictive Path Stressing - Inference (PPS-I).
    G: Đồ thị NetworkX.
    K: Tập hợp các cặp (s, t) quan trọng.
    T: Ngưỡng chi phí.
    edge_cost_functions: Dictionary các hàm chi phí cạnh, key là edge_id (u,v).
    b_budgets: Dictionary các giới hạn ngân sách tối đa cho mỗi cạnh, key là edge_id (u,v).
    x_initial: Vector nhiễu ban đầu (dictionary edge_id -> giá trị nhiễu).
    soft_epsilon: Tham số cho ngưỡng mềm của hàm tiềm năng.
    """
    x = x_initial.copy()

    # Bước 3: Xác định các cặp vi phạm ban đầu.
    # Trong PPS-I, chúng ta sử dụng Dijkstra và chi phí thực tế (f_e).
    K_violate = get_violating_pairs(G, K, T, x, edge_cost_functions, use_spagan=False)

    while K_violate:
        P_paths = []  # Tập hợp các đường đi ngắn nhất cho các cặp vi phạm
        for s, t in K_violate:
            # Trong PPS-I, sử dụng DijkstraPath với chi phí thực tế (f_e).
            _, rho_s_t = dijkstra_path_cost(G, s, t, x, edge_cost_functions)
            if rho_s_t:
                P_paths.append(rho_s_t)

        current_C_Px = compute_soft_potential_function(P_paths, x, T, G=G, edge_cost_functions=edge_cost_functions,
                                                       use_spagan=False)

        while current_C_Px < len(P_paths) * T - soft_epsilon:
            e_star, delta_star, _ = find_optimal_increase(G, P_paths, x, T, b_budgets, soft_epsilon,
                                                          edge_cost_functions=edge_cost_functions, G=G,
                                                          use_spagan=False)

            if e_star is None or delta_star == 0:  # Không tìm thấy cải thiện nào
                break

            x[e_star] = x.get(e_star, 0) + delta_star
            current_C_Px = compute_soft_potential_function(P_paths, x, T, G=G, edge_cost_functions=edge_cost_functions,
                                                           use_spagan=False)

        # Bước 27: Cập nhật các cặp vi phạm bằng cách tính toán lại đường đi ngắn nhất chính xác.
        K_violate = get_violating_pairs(G, K, T, x, edge_cost_functions, use_spagan=False)

    return x


# --- 5. Thuật toán 6: Refine (Mô phỏng) ---
# Giai đoạn này huấn luyện tác nhân RL và tinh chỉnh giải pháp.

def refine_phase(q_ebm, omega_mix_cvae, D_sol, b_budgets, K_max_rl_episodes=5):
    """
    Mô phỏng giai đoạn Refine.
    q_ebm: Mô hình EBM từ giai đoạn Morph.
    omega_mix_cvae: Mô hình Mix-CVAE từ giai đoạn Morph.
    D_sol: Tập dữ liệu giải pháp ban đầu, được cập nhật liên tục.
    K_max_rl_episodes: Số lượng episode huấn luyện RL.
    """
    print("\n--- Bắt đầu giai đoạn Refine (Mô phỏng) ---")
    # Tác nhân RL (policy pi) sẽ học để khám phá không gian tiềm ẩn của Mix-CVAE.
    # Hàm reward sẽ được thiết kế để khuyến khích các giải pháp khả thi và chi phí thấp.

    # Mô phỏng RL agent (pi)
    mock_rl_policy = lambda state, context: {'mu': np.random.rand(128),
                                             'sigma': np.random.rand(128)}  # Trả về hành động (mu, sigma)

    S_new = []  # Lưu trữ các giải pháp được tinh chỉnh trong episode này

    for episode in range(K_max_rl_episodes):
        # Lấy mẫu một instance (G, K, T, x) từ D_sol
        if not D_sol:
            print("D_sol is empty, skipping Refine phase updates.")
            break
        selected_sol = random.choice(D_sol)
        G_sampled = selected_sol['graph']
        K_sampled = selected_sol['K']
        T_sampled = selected_sol['T']
        x_sampled = selected_sol['x_sol']
        # Bạn cũng cần edge_cost_functions cho c_context
        edge_cost_functions_sampled = selected_sol['edge_cost_functions']
        c_context = {
            'graph': G_sampled,
            'K': K_sampled,
            'T': T_sampled,
            'edges': list(G_sampled.edges()),
            'edge_cost_functions': edge_cost_functions_sampled  # Thêm cái này
        }

        # Encode giải pháp hiện tại vào không gian tiềm ẩn
        # (Trong thực tế, thông qua encoder của Mix-CVAE)
        current_latent_z = np.random.rand(128)  # Mô phỏng latent vector

        max_steps_per_episode = 10  # T_max
        for step in range(max_steps_per_episode):
            # RL agent samples action (mu, sigma)
            action = mock_rl_policy(current_latent_z, c_context)
            mu, sigma = action['mu'], action['sigma']

            # Áp dụng nhiễu stochastic vào latent vector
            delta = sigma * np.random.randn(128) + mu
            next_latent_z = current_latent_z + delta

            # Giải mã latent vector mới thành vector nhiễu x
            x_raw = omega_mix_cvae.decode(next_latent_z, c_context)  #

            # Tính toán phần thưởng (reward)
            # Hàm reward phức tạp R(x) = F(G,K,x) - kappa * log(1 + ||x||_1)
            # F(G,K,x) là tổng điểm khả thi (sigmoid-based term)
            # ||x||_1 là chuẩn L1 của x

            # Để đơn giản, mô phỏng phần thưởng:
            # Reward cao nếu x_raw đạt ngưỡng T cho các cặp K_sampled
            # và tổng nhiễu x_raw không quá lớn.
            is_feasible = True
            current_total_perturbation = sum(x_raw.values())
            for s, t in K_sampled:
                cost, _ = dijkstra_path_cost(G_sampled, s, t, x_raw, c_context[
                    'edge_cost_functions'])  # Giả sử có edge_cost_functions trong context
                if cost < T_sampled:
                    is_feasible = False
                    break

            reward = 0
            if is_feasible:
                reward += 100  # Phần thưởng cao cho tính khả thi
                reward -= current_total_perturbation * 0.1  # Phạt cho chi phí cao
            else:
                reward -= 10  # Phạt cho không khả thi

            # Store transition in replay buffer (mô phỏng)
            # (current_latent_z, c_context, action, reward, next_latent_z, c_context)

            # Update policy (mô phỏng)
            # (e.g., PPO, DDPG, gradient-ascent)

            current_latent_z = next_latent_z

            # Nếu reward đủ cao hoặc đạt max_steps, tinh chỉnh và thêm vào S_new
            if reward >= 50 or step == max_steps_per_episode - 1:  # Ngưỡng reward_thresh
                # PPS-I để đảm bảo 100% tính khả thi
                refined_x = predictive_path_stressing_inference(G_sampled, K_sampled, T_sampled,
                                                                c_context['edge_cost_functions'], b_budgets, x_raw)
                S_new.append({'graph': G_sampled, 'K': K_sampled, 'T': T_sampled, 'x_refined': refined_x})
                break

        # Sau mỗi E_retrain_freq episode, retrain Morph (mô phỏng)
        if (episode + 1) % 5 == 0:  # Ví dụ: mỗi 5 episode
            print(f"Retraining Morph after episode {episode + 1}...")
            # D_sol_augmented = D_sol + S_new
            # q_ebm, omega_mix_cvae = morph_phase(D_sol_augmented) # Gọi lại Morph
            # Để tránh vòng lặp vô hạn, chỉ in thông báo. Trong thực tế sẽ gọi morph_phase.

    # Sắp xếp các giải pháp trong S_new và chọn top K
    S_new.sort(key=lambda item: sum(item['x_refined'].values()))  # Sắp xếp theo tổng nhiễu
    S_topK = S_new[:10]  # Top 10 giải pháp

    # Bổ sung D_sol với Top-K Solutions
    D_sol.extend([item for item in S_topK if item not in D_sol])  # Chỉ thêm các giải pháp mới

    print("--- Kết thúc giai đoạn Refine ---")
    return S_topK, q_ebm, omega_mix_cvae, mock_rl_policy  # Trả về các mô hình đã tinh chỉnh và các giải pháp topK


# --- 6. Thuật toán 7: Inference Process ---
# Quá trình suy luận cuối cùng để tạo giải pháp gần tối ưu.

def inference_process(G_new, K_new, T_new, trained_omega, trained_pi, trained_F_theta, b_budgets, edge_cost_functions):
    print("\n--- Bắt đầu Quy trình Suy luận (Inference Process) ---")

    # Định nghĩa input_dim và condition_dim cho inference
    # Cần phải nhất quán với cách CVAE được khởi tạo trong Morph
    input_dim_cvae = len(G_new.edges())  # Giả sử đồ thị mới có số cạnh giống hoặc ít hơn
    condition_dim_cvae = 5  # Giả sử giống với Morph

    # Chuẩn bị điều kiện c_new cho CVAE
    c_new_dict = {
        'graph': G_new,
        'K': K_new,
        'T': T_new,
        'edges': list(G_new.edges()),  # Cần để decode biết các cạnh
        'edge_cost_functions': edge_cost_functions  # Để PPS-I sử dụng
    }
    # Mã hóa c_new thành vector (mô phỏng)
    c_new_vec = np.random.rand(condition_dim_cvae)  # Điều kiện phức tạp sẽ được mã hóa

    # Lấy mẫu latent vector ban đầu (z_init) từ prior của Mix-CVAE.
    z_init = trained_omega.sample_latent(c_new_vec)  # Sử dụng c_new_vec
    z_star = z_init

    K_inference_steps = 20  # Số bước suy luận
    for step in range(K_inference_steps):
        # RL policy hành động trên trạng thái hiện tại.
        # State = (latent_z, c_vec)
        action = trained_pi(z_star, c_new_vec)  # Giả định trained_pi cũng nhận c_vec
        mu, sigma = action['mu'], action['sigma']

        # Áp dụng nhiễu Gaussian vào latent vector.
        delta = sigma * np.random.randn(trained_omega.latent_dim) + mu  # Kích thước delta phải khớp latent_dim
        z_star = z_star + delta

    # Giải mã latent vector thành vector nhiễu thô (x_raw).
    x_raw = trained_omega.decode(z_star, c_new_dict)  # Truyền c_new_dict để decode có thể xác định các cạnh

    # Tinh chỉnh cuối cùng để đảm bảo tính khả thi 100% bằng PPS-I.
    x_final_star = predictive_path_stressing_inference(G_new, K_new, T_new, edge_cost_functions, b_budgets, x_raw)

    print("--- Kết thúc Quy trình Suy luận ---")
    return x_final_star


# --- 7. Thuật toán 1: Hephaestus Main Framework (Điều phối tổng thể) ---

def hephaestus_main_framework(D_graph, K_critical_pairs, T_thresholds, max_forge_iter=1,
                              max_morph_iter=100, max_refine_episodes=5,
                              initial_experts=1, b_budgets_default_value=25):
    """
    Framework chính của Hephaestus, điều phối các giai đoạn Forge, Morph và Refine.
    D_graph: Tập dữ liệu đồ thị ban đầu (ví dụ: danh sách các đồ thị).
    K_critical_pairs: Tập hợp các cặp (s,t) quan trọng.
    T_thresholds: Tập hợp các ngưỡng T.
    """
    print("\n========== Bắt đầu Framework Hephaestus ==========")
    D_sol = []  # Tập hợp các giải pháp ban đầu, được khởi tạo trống

    # Phase 1: Forge (Tạo tập dữ liệu giải pháp ban đầu)
    # Thuật toán 2: Forge
    # Thuật toán 3: SPAGAN Training (Mô phỏng)
    # Thuật toán 4: Predictive Path Stressing (PPS)
    print("\n--- Giai đoạn 1: Forge ---")
    F_theta_trained = mock_spagan_model  # SPAGAN được huấn luyện

    for G_i in D_graph:
        # Giả sử hàm chi phí tuyến tính cho mục đích Forge
        current_edge_cost_functions = {edge: linear_edge_cost for edge in G_i.edges()}
        initial_x = {edge: 0 for edge in G_i.edges()}
        b_budgets = {edge: b_budgets_default_value for edge in G_i.edges()}

        # Với mỗi đồ thị, cặp K và ngưỡng T, chạy PPS
        for T_val in T_thresholds:
            print(f"  Forge: Xử lý đồ thị {G_i.graph['id']} với ngưỡng T={T_val}...")
            x_perturbation = predictive_path_stressing(G_i, K_critical_pairs, T_val,
                                                       current_edge_cost_functions, b_budgets,
                                                       initial_x, F_theta_trained)  #
            D_sol.append({'graph': G_i, 'K': K_critical_pairs, 'T': T_val, 'x_sol': x_perturbation,
                          'edge_cost_functions': current_edge_cost_functions})
            print(
                f"  Forge: Đã tạo giải pháp cho đồ thị {G_i.graph['id']}, T={T_val}. Total perturbations: {sum(x_perturbation.values())}")

    # Phase 2: Morph (Mô hình hóa phân phối giải pháp)
    # Thuật toán 5: Morph
    q_ebm_trained, omega_mix_cvae_trained = morph_phase(D_sol, K_max_morph_iter=max_morph_iter,
                                                        N_initial_experts=initial_experts)  #

    # Phase 3: Refine (Tối ưu hóa chính sách RL)
    # Thuật toán 6: Refine
    # Thuật toán 8: Predictive Path Stressing - Inference (PPS-I)
    print("\n--- Giai đoạn 3: Refine ---")
    top_k_solutions_refine, q_ebm_trained, omega_mix_cvae_trained, pi_rl_trained = \
        refine_phase(q_ebm_trained, omega_mix_cvae_trained, D_sol,b_budgets_default_value, K_max_rl_episodes=max_refine_episodes)  #

    print("\n========== Kết thúc Framework Hephaestus ==========")
    return F_theta_trained, q_ebm_trained, omega_mix_cvae_trained, pi_rl_trained


# --- Ví dụ sử dụng toàn bộ Framework ---

if __name__ == "__main__":
    # Tạo một đồ thị ví dụ
    G1 = nx.DiGraph(id='G1')
    G1.add_edges_from([(1, 2, {'base_weight': 1}), (2, 3, {'base_weight': 1}), (1, 3, {'base_weight': 3}),
                       (3, 4, {'base_weight': 1}), (4, 5, {'base_weight': 1}), (3, 5, {'base_weight': 3})])

    G2 = nx.DiGraph(id='G2')
    G2.add_edges_from([(1, 2, {'base_weight': 2}), (2, 4, {'base_weight': 2}), (1, 3, {'base_weight': 5}),
                       (3, 4, {'base_weight': 1}), (4, 5, {'base_weight': 1})])

    D_graph_data = [G1, G2]  # Tập dữ liệu đồ thị ban đầu
    K_critical = [(1, 3), (3, 5)]  # Các cặp nguồn-đích quan trọng
    T_threshold_values = [4, 5]  # Các ngưỡng chi phí

    # Chạy framework Hephaestus
    # Kết quả trả về là các mô hình đã huấn luyện/mô phỏng
    trained_spagan, trained_ebm, trained_mix_cvae, trained_rl_policy = \
        hephaestus_main_framework(D_graph_data, K_critical, T_threshold_values,
                                  max_forge_iter=1, max_morph_iter=50, max_refine_episodes=5)

    # --- Thực hiện suy luận với một instance mới ---
    # Thuật toán 7: Inference Process
    print("\n--- Thực hiện Suy luận với instance mới ---")
    G_test = nx.DiGraph(id='G_test')
    G_test.add_edges_from([(10, 11, {'base_weight': 1}), (11, 12, {'base_weight': 1}), (10, 12, {'base_weight': 4})])
    K_test = [(10, 12)]
    T_test = 5
    b_budgets_default_value = 25  # Hoặc 15, 20... một giá trị đủ lớn để thoát vòng lặp

    # Giả định các hàm chi phí cạnh cho đồ thị test
    test_edge_cost_functions = {edge: linear_edge_cost for edge in G_test.edges()}
    test_b_budgets = {edge: 5 for edge in G_test.edges()}

    x_final_solution = inference_process(G_test, K_test, T_test,
                                         trained_mix_cvae, trained_rl_policy, trained_spagan,
                                         test_b_budgets, test_edge_cost_functions)  #
    print(f"Giải pháp nhiễu cuối cùng cho instance test: {x_final_solution}")

    # Kiểm tra lại chi phí đường đi cho instance test sau suy luận
    print("\nKiểm tra chi phí đường đi sau Suy luận:")
    for s, t in K_test:
        cost, path = dijkstra_path_cost(G_test, s, t, x_final_solution, test_edge_cost_functions)
        print(f"Từ {s} đến {t}: Chi phí = {cost}, Đường đi = {path}, Đạt ngưỡng ({cost >= T_test})")