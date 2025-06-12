import networkx as nx
import numpy as np

# --- 1. Hàm hỗ trợ chung ---

def dijkstra_path_cost(G, s, t, x_perturbation, edge_cost_functions):
    """
    Tính toán chi phí đường đi ngắn nhất từ s đến t trên đồ thị G,
    sử dụng trọng số cạnh được điều chỉnh bởi x_perturbation và edge_cost_functions.
    Trả về chi phí và đường đi.
    """
    # Tạo một đồ thị tạm thời với trọng số cạnh được điều chỉnh
    G_temp = nx.DiGraph()
    G_temp.add_nodes_from(G.nodes)
    for u, v, data in G.edges(data=True):
        edge_id = (u, v)
        # Giả sử edge_id là một tuple (u, v)
        # Chi phí của cạnh được xác định bởi hàm chi phí cạnh và giá trị nhiễu x_perturbation
        # Nếu x_perturbation không chứa edge_id, giả sử giá trị nhiễu là 0
        perturb_value = x_perturbation.get(edge_id, 0)
        cost = edge_cost_functions[edge_id](perturb_value) if edge_id in edge_cost_functions else 1.0 # Default cost
        G_temp.add_edge(u, v, weight=cost)

    try:
        path = nx.dijkstra_path(G_temp, source=s, target=t, weight='weight')
        cost = nx.dijkstra_path_length(G_temp, source=s, target=t, weight='weight')
        return cost, path
    except nx.NetworkXNoPath:
        return float('inf'), []
        # Trả về vô cùng nếu không có đường đi

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
            # Trong một triển khai thực tế, F_theta sẽ là một mô hình ML đã huấn luyện.
            # spagan_model(G, s, t, x_perturbation) sẽ là một hàm ước tính.
            estimated_cost = spagan_model(G, s, t, x_perturbation)
            if estimated_cost < T:
                violating_pairs.add((s, t))
        else:
            cost, _ = dijkstra_path_cost(G, s, t, x_perturbation, edge_cost_functions)
            if cost < T:
                violating_pairs.add((s, t))
    return violating_pairs

def compute_soft_potential_function(P, x_perturbation, T, spagan_model=None, G=None, edge_cost_functions=None, use_spagan=False):
    """
    Tính toán hàm tiềm năng mềm C(P,x).
    P là tập hợp các đường đi ngắn nhất đã tìm thấy cho các cặp vi phạm.
    """
    C_Px = 0
    for path_s_t in P:
        s, t = path_s_t[0], path_s_t[-1]
        # Lấy điểm nguồn và đích từ đường đi
        if use_spagan:
            # spagan_model cần ước tính chi phí cho một cặp (s,t) và x
            cost_st = spagan_model(G, s, t, x_perturbation)
        else:
            cost_st, _ = dijkstra_path_cost(G, s, t, x_perturbation, edge_cost_functions)
        C_Px += min(T, cost_st)
    return C_Px


def find_optimal_increase(G, P, x_perturbation, T, b_budgets, soft_epsilon, spagan_model=None, edge_cost_functions=None,
                          use_spagan=False):
    """
    Tìm cạnh e* và mức tăng Delta* dẫn đến mức tăng tiềm năng lớn nhất trên một đơn vị ngân sách.
    """
    E_P = set()
    # Tập hợp các cạnh trong tất cả các đường đi trong P
    for path_s_t in P:
        for i in range(len(path_s_t) - 1):
            E_P.add((path_s_t[i], path_s_t[i + 1]))

    e_star = None
    delta_star = 0
    max_delta_ratio = -float('inf')

    C_Px_current = compute_soft_potential_function(P, x_perturbation, T, spagan_model, G, edge_cost_functions,
                                                   use_spagan)

    for e in E_P:
        u, v = e
        # Giả sử b_budgets là một dictionary mapping edge_id to its max budget
        max_delta_e = b_budgets.get(e, 0) - x_perturbation.get(e, 0)

        for delta in range(1, int(max_delta_e) + 1):
            # Delta từ 1 đến max_delta_e
            x_prime = x_perturbation.copy()
            x_prime[e] = x_prime.get(e, 0) + delta

            C_Px_prime = compute_soft_potential_function(P, x_prime, T, spagan_model, G, edge_cost_functions,
                                                         use_spagan)

            if delta > 0:
            # Tránh chia cho 0
                delta_ratio = (C_Px_prime - C_Px_current) / delta
                if delta_ratio > max_delta_ratio:
                    max_delta_ratio = delta_ratio
                    e_star = e
                    delta_star = delta

    return e_star, delta_star, max_delta_ratio


# --- 2. Mô phỏng SPAGAN và hàm chi phí cạnh ---

def mock_spagan_model(G, s, t, x_perturbation):
    """
    Mô phỏng mô hình SPAGAN (F_theta).
    Trong thực tế, đây sẽ là một mô hình ML phức tạp.
    Ở đây, chúng ta chỉ giả định nó trả về một chi phí ước tính dựa trên độ dài đường đi
    và một phần nào đó của x_perturbation.
    """
    # Một cách đơn giản để mô phỏng: trả về chi phí đường đi ngắn nhất Euclidean
    # cộng với một "chi phí nhiễu" đơn giản.

    # Để đơn giản, giả sử mỗi cạnh có trọng số cơ bản là 1, và nhiễu cộng thêm.
    # Trong mô hình SPAGAN thật, nó sẽ học cách ước tính.

    # Đây là nơi SPAGAN sẽ ước tính chi phí. Chúng ta sẽ làm một ước tính heuristic.
    # Giả sử G đã có 'weight' trên các cạnh, hoặc mặc định là 1.
    G_weighted = nx.DiGraph()
    G_weighted.add_nodes_from(G.nodes)
    for u, v, data in G.edges(data=True):
        base_weight = data.get('weight', 1.0)
        perturb_value = x_perturbation.get((u, v), 0)
        # SPAGAN sẽ ước tính tác động của perturb_value lên chi phí đường đi.
        # Ví dụ: base_weight + perturb_value * some_factor
        estimated_cost = base_weight + perturb_value * 0.1  # Đây chỉ là một ví dụ
        G_weighted.add_edge(u, v, weight=estimated_cost)

    try:
        return nx.dijkstra_path_length(G_weighted, s, t, weight='weight')
    except nx.NetworkXNoPath:
        return float('inf')

def linear_edge_cost(x_e):
    """Hàm chi phí cạnh tuyến tính: f_e(x_e) = x_e"""
    return x_e

def quadratic_edge_cost(x_e):
    """Hàm chi phí cạnh bậc hai: f_e(x_e) = x_e^2"""
    return x_e ** 2

def log_concave_edge_cost(x_e):
    """Hàm chi phí cạnh log-concave: f_e(x_e) = log(1 + x_e)"""
    return np.log(1 + x_e) if x_e >= 0 else -np.inf # log không xác định cho x_e < 0


# --- 3. Thuật toán PPS (Predictive Path Stressing) ---

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

    # Bước 2 & 3: Xác định các cặp vi phạm
    K_violate = get_violating_pairs(G, K, T, x, edge_cost_functions, spagan_model=F_theta, use_spagan=True)

    while K_violate:
        P = []  # Tập hợp các đường đi ngắn nhất cho các cặp vi phạm
        for s, t in K_violate:
            # SPAGANPath: trong PPS, SPAGAN được dùng để tìm đường đi và ước tính chi phí.
            # Tuy nhiên, để tìm đường đi ngắn nhất với SPAGAN, cần một hàm phức tạp hơn.
            # Để đơn giản hóa mô phỏng, chúng ta sẽ sử dụng Dijkstra với chi phí ước tính
            # từ SPAGAN để xác định đường đi, và sau đó tính toán chi phí thực tế.

            # SPAGANPath (G, s, t, x): lấy đường đi ngắn nhất được ước tính bởi SPAGAN
            # Đối với mục đích mô phỏng, chúng ta sẽ giả định SPAGANPath trả về một đường đi.
            # Trong thực tế, SPAGAN sẽ đưa ra ước tính và dựa vào đó để xác định đường đi.
            # Để đơn giản, tôi sẽ dùng Dijkstra với trọng số từ SPAGAN để có đường đi.

            # Tạo đồ thị với trọng số ước tính từ SPAGAN
            G_spagan_weighted = nx.DiGraph()
            G_spagan_weighted.add_nodes_from(G.nodes)
            for u, v, data in G.edges(data=True):
                edge_id = (u, v)
                estimated_cost = F_theta(G, u, v, x)
                # SPAGAN ước tính chi phí cho cạnh (u,v)
                G_spagan_weighted.add_edge(u, v, weight=estimated_cost)

            try:
                rho_s_t = nx.dijkstra_path(G_spagan_weighted, s, t, weight='weight')
            except nx.NetworkXNoPath:
                rho_s_t = []  # Không có đường đi

            if rho_s_t:
                P.append(rho_s_t)

        # Bước 9-24: Đánh giá hàm tiềm năng mềm và cập nhật x
        current_C_Px = compute_soft_potential_function(P, x, T, spagan_model=F_theta, G=G, use_spagan=True)

        # while C(P,x) < |P|*T - soft_epsilon do
        # Đây là phần phức tạp nhất, cần lặp lại cho đến khi hàm tiềm năng đạt ngưỡng.
        # Chúng ta sẽ thực hiện việc chọn cạnh và tăng giá trị.
        while current_C_Px < len(P) * T - soft_epsilon:
            e_star, delta_star, _ = find_optimal_increase(G, P, x, T, b_budgets, soft_epsilon, spagan_model=F_theta,
                                                          G=G, use_spagan=True)

            if e_star is None:  # Không tìm thấy cải thiện nào
                break

            x[e_star] = x.get(e_star, 0) + delta_star
            current_C_Px = compute_soft_potential_function(P, x, T, spagan_model=F_theta, G=G, use_spagan=True)

        # Bước 25: Cập nhật các cặp vi phạm
        K_violate = get_violating_pairs(G, K, T, x, edge_cost_functions, spagan_model=F_theta, use_spagan=True)

    return x


# --- 4. Thuật toán PPS-I (Predictive Path Stressing - Inference) ---

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

    # Bước 3: Xác định các cặp vi phạm ban đầu
    # Trong PPS-I, chúng ta sử dụng Dijkstra và chi phí thực tế (f_e)
    violating_pairs = set()
    for s, t in K:
        cost, _ = dijkstra_path_cost(G, s, t, x, edge_cost_functions)
        if cost < T:
            violating_pairs.add((s, t))
    K_violate = violating_pairs

    while K_violate:
        P = []  # Tập hợp các đường đi ngắn nhất cho các cặp vi phạm
        for s, t in K_violate:
            # Trong PPS-I, sử dụng DijkstraPath với chi phí thực tế (f_e)
            _, rho_s_t = dijkstra_path_cost(G, s, t, x, edge_cost_functions)
            if rho_s_t:
                P.append(rho_s_t)

        current_C_Px = compute_soft_potential_function(P, x, T, G=G, edge_cost_functions=edge_cost_functions,
                                                       use_spagan=False)

        while current_C_Px < len(P) * T - soft_epsilon:
            e_star, delta_star, _ = find_optimal_increase(G, P, x, T, b_budgets, soft_epsilon,
                                                          edge_cost_functions=edge_cost_functions, G=G,
                                                          use_spagan=False)

            if e_star is None:  # Không tìm thấy cải thiện nào
                break

            x[e_star] = x.get(e_star, 0) + delta_star
            current_C_Px = compute_soft_potential_function(P, x, T, G=G, edge_cost_functions=edge_cost_functions,
                                                           use_spagan=False)

        # Bước 27: Cập nhật các cặp vi phạm bằng cách tính toán lại đường đi ngắn nhất chính xác
        new_violating_pairs = set()
        for s, t in K:
            cost, _ = dijkstra_path_cost(G, s, t, x, edge_cost_functions)
            if cost < T:
                new_violating_pairs.add((s, t))
        K_violate = new_violating_pairs

    return x


# --- Ví dụ sử dụng ---

if __name__ == "__main__":
    # Tạo một đồ thị ví dụ
    G = nx.DiGraph()
    G.add_edges_from([(1, 2, {'weight': 1}), (2, 3, {'weight': 1}), (1, 3, {'weight': 3}),
                      (3, 4, {'weight': 1}), (4, 5, {'weight': 1}), (3, 5, {'weight': 3})])

    # Các cặp điểm nguồn-đích quan trọng
    K = [(1, 3), (3, 5)]

    # Ngưỡng chi phí
    T = 4

    # Các hàm chi phí cạnh
    # Chúng ta định nghĩa các hàm này cho từng cạnh cụ thể hoặc một hàm mặc định.
    # Để đơn giản, giả sử tất cả các cạnh sử dụng hàm chi phí tuyến tính
    edge_cost_functions = {
        (u, v): linear_edge_cost for u, v, _ in G.edges(data=True)
    }

    # Ngân sách tối đa cho mỗi cạnh (ví dụ: mỗi cạnh có thể tăng nhiễu tối đa là 5)
    b_budgets = {edge: 5 for edge in G.edges()}

    # Vector nhiễu ban đầu (tất cả là 0)
    x_initial = {edge: 0 for edge in G.edges()}

    print("--- Chạy Predictive Path Stressing (PPS) ---")
    # Sử dụng mock_spagan_model làm F_theta
    final_x_pps = predictive_path_stressing(G, K, T, edge_cost_functions, b_budgets, x_initial, mock_spagan_model)
    print(f"Vector nhiễu cuối cùng (PPS): {final_x_pps}")

    # Kiểm tra chi phí đường đi sau PPS
    print("\nChi phí đường đi sau PPS (sử dụng Dijkstra và chi phí thực tế):")
    for s, t in K:
        cost, path = dijkstra_path_cost(G, s, t, final_x_pps, edge_cost_functions)
        print(f"Từ {s} đến {t}: Chi phí = {cost}, Đường đi = {path}, Đạt ngưỡng ({cost >= T})")

    print("\n--- Chạy Predictive Path Stressing - Inference (PPS-I) ---")
    final_x_ppsi = predictive_path_stressing_inference(G, K, T, edge_cost_functions, b_budgets, x_initial)
    print(f"Vector nhiễu cuối cùng (PPS-I): {final_x_ppsi}")

    # Kiểm tra chi phí đường đi sau PPS-I
    print("\nChi phí đường đi sau PPS-I (sử dụng Dijkstra và chi phí thực tế):")
    for s, t in K:
        cost, path = dijkstra_path_cost(G, s, t, final_x_ppsi, edge_cost_functions)
        print(f"Từ {s} đến {t}: Chi phí = {cost}, Đường đi = {path}, Đạt ngưỡng ({cost >= T})")

    print("\n--- Ví dụ với hàm chi phí bậc hai (PPS-I) ---")
    quadratic_edge_cost_functions = {
        (u, v): quadratic_edge_cost for u, v, _ in G.edges(data=True)
    }
    final_x_ppsi_quad = predictive_path_stressing_inference(G, K, T, quadratic_edge_cost_functions, b_budgets,
                                                            x_initial)
    print(f"Vector nhiễu cuối cùng (PPS-I, Quadratic): {final_x_ppsi_quad}")
    print("\nChi phí đường đi sau PPS-I (Quadratic, sử dụng Dijkstra và chi phí thực tế):")
    for s, t in K:
        cost, path = dijkstra_path_cost(G, s, t, final_x_ppsi_quad, quadratic_edge_cost_functions)
        print(f"Từ {s} đến {t}: Chi phí = {cost}, Đường đi = {path}, Đạt ngưỡng ({cost >= T})")

    print("\n--- Ví dụ với hàm chi phí log-concave (PPS-I) ---")
    log_concave_edge_cost_functions = {
        (u, v): log_concave_edge_cost for u, v, _ in G.edges(data=True)
    }
    final_x_ppsi_log = predictive_path_stressing_inference(G, K, T, log_concave_edge_cost_functions, b_budgets,
                                                           x_initial)
    print(f"Vector nhiễu cuối cùng (PPS-I, Log-Concave): {final_x_ppsi_log}")
    print("\nChi phí đường đi sau PPS-I (Log-Concave, sử dụng Dijkstra và chi phí thực tế):")
    for s, t in K:
        cost, path = dijkstra_path_cost(G, s, t, final_x_ppsi_log, log_concave_edge_cost_functions)
        print(f"Từ {s} đến {t}: Chi phí = {cost}, Đường đi = {path}, Đạt ngưỡng ({cost >= T})")

