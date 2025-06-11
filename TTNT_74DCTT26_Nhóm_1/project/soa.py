import numpy as np
import matplotlib.pyplot as plt

# Các thông số thuật toán
NUM_SEAGULLS = 20
NUM_ITERATIONS = 100
MIGRATION_FACTOR = 0.3  # Xác suất di chuyển ban đầu
BETA = 0.7  # Hệ số xoắn ốc ban đầu
SPIRAL_FACTOR = 1.5
CONVERGENCE_THRESHOLD = 1e-5  # Ngưỡng hội tụ
MAX_NO_IMPROVE = 20  # Số lần lặp không cải thiện để dừng

# Danh sách các thành phố (tọa độ x, y)
cities = np.array([
    [0, 0], [1, 5], [5, 2], [6, 6], [8, 3], [7, 9], [2, 7], [3, 3]
])
num_cities = len(cities)

# Ma trận khoảng cách giữa các thành phố
distance_matrix = np.linalg.norm(cities[:, np.newaxis] - cities[np.newaxis, :], axis=2)

def tinh_quang_duong(path):
    """ Tính tổng quãng đường của một hành trình """
    return np.sum([distance_matrix[path[i], path[i+1]] for i in range(len(path) - 1)])

def two_opt(path):
    """ Tối ưu hóa lộ trình bằng thuật toán 2-opt """
    best_path = path.copy()
    best_distance = tinh_quang_duong(best_path)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                new_distance = tinh_quang_duong(new_path)
                if new_distance < best_distance:
                    best_path, best_distance = new_path, new_distance
                    improved = True
    return best_path

# Khởi tạo tập hợp hải âu
seagulls = [np.random.permutation(num_cities).tolist() + [np.random.permutation(num_cities)[0]] for _ in range(NUM_SEAGULLS)]
global_best_path = min(seagulls, key=tinh_quang_duong)
global_best_distance = tinh_quang_duong(global_best_path)
no_improve_count = 0
history = [global_best_distance]

for iteration in range(NUM_ITERATIONS):
    new_seagulls = []
    migration_factor = MIGRATION_FACTOR * (1 - iteration / NUM_ITERATIONS)
    beta = BETA * (1 - iteration / NUM_ITERATIONS)

    for seagull in seagulls:
        new_path = seagull[:]
        
        # Giai đoạn di chuyển (Migration)
        if np.random.rand() < migration_factor:
            swap_idx = np.random.choice(range(num_cities), 2, replace=False)
            new_path[swap_idx[0]], new_path[swap_idx[1]] = new_path[swap_idx[1]], new_path[swap_idx[0]]

        # Giai đoạn tấn công xoắn ốc (Spiral Attack)
        center = np.mean(cities[new_path[:-1]], axis=0)
        spiral_positions = cities[new_path[:-1]] + beta * (cities[new_path[:-1]] - center)
        spiral_positions = np.clip(spiral_positions, np.min(cities), np.max(cities))
        
        # Sắp xếp lại hành trình dựa trên vị trí mới
        spiral_distances = np.linalg.norm(spiral_positions[:, np.newaxis] - cities[np.newaxis, :], axis=2)
        new_order = np.argsort(spiral_distances.sum(axis=1)).tolist() + [np.argsort(spiral_distances.sum(axis=1))[0]]

        # Áp dụng tối ưu hóa 2-opt
        new_order = two_opt(new_order)
        new_seagulls.append(new_order)

    seagulls = new_seagulls
    best_path = min(seagulls, key=tinh_quang_duong)
    best_distance = tinh_quang_duong(best_path)

    # Cập nhật lộ trình tốt nhất toàn cục
    if best_distance < global_best_distance:
        global_best_path, global_best_distance = best_path, best_distance
        no_improve_count = 0
    else:
        no_improve_count += 1

    history.append(global_best_distance)
    print(f"Lần lặp {iteration + 1}, Quãng đường ngắn nhất: {global_best_distance:.2f}")

    # Dừng sớm nếu không cải thiện
    if no_improve_count >= MAX_NO_IMPROVE:
        print("Hội tụ sớm, dừng thuật toán!")
        break

# Vẽ biểu đồ hội tụ
plt.figure(figsize=(10, 4))
plt.plot(history, 'b-', linewidth=2)
plt.title("Biểu đồ hội tụ của quãng đường tối ưu")
plt.xlabel("Lần lặp")
plt.ylabel("Quãng đường")
plt.grid(True)
plt.show()

# Vẽ kết quả tối ưu hóa với mũi tên hướng bay
plt.figure(figsize=(10, 8))

# Vẽ các thành phố
for i in range(num_cities):
    plt.plot(cities[i][0], cities[i][1], 'ro', markersize=10)
    plt.text(cities[i][0] + 0.1, cities[i][1] + 0.1, str(i), fontsize=12)

# Vẽ đường đi và mũi tên
for i in range(len(global_best_path) - 1):
    a, b = global_best_path[i], global_best_path[i+1]
    x_start, y_start = cities[a]
    x_end, y_end = cities[b]
    
    # Vẽ đường nối
    plt.plot([x_start, x_end], [y_start, y_end], 'b-', linewidth=2, alpha=0.8)
    
    # Vẽ mũi tên biểu diễn hướng bay
    dx = x_end - x_start
    dy = y_end - y_start
    plt.arrow(x_start, y_start, dx * 0.85, dy * 0.85,  # rút ngắn mũi tên để không đè lên điểm
              head_width=0.3, head_length=0.4, fc='blue', ec='blue', length_includes_head=True)

plt.title(f"SOA: Quãng đường tối ưu = {global_best_distance:.2f}")
plt.grid(True)
plt.axis("equal")
plt.show()