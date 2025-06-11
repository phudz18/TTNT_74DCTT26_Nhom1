import numpy as np
import matplotlib.pyplot as plt

NUM_ANTS = 10
NUM_ITERATIONS = 100
ALPHA, BETA, EVAPORATION, Q = 1.0, 5.0, 0.5, 100

cities = np.array([
    [0, 0], [1, 5], [5, 2], [6, 6], [8, 3], [7, 9], [2, 7], [3, 3]
])
num_cities = len(cities)

distance_matrix = np.linalg.norm(cities[:, np.newaxis] - cities[np.newaxis, :], axis=2)
pheromone = np.ones((num_cities, num_cities))

def chon_thanh_pho_tiep_theo(current_city, visited):
    unvisited = list(set(range(num_cities)) - set(visited))
    probabilities = (pheromone[current_city, unvisited]**ALPHA) * ((1.0 / distance_matrix[current_city, unvisited])**BETA)
    probabilities /= np.sum(probabilities) if np.sum(probabilities) else np.ones(len(unvisited)) / len(unvisited)
    return np.random.choice(unvisited, p=probabilities)

best_distance = float('inf')
best_path = []
all_paths_history = []

for _ in range(NUM_ITERATIONS):
    all_paths, all_lengths = [], []

    for _ in range(NUM_ANTS):
        path = [np.random.randint(num_cities)]
        while len(path) < num_cities:
            path.append(chon_thanh_pho_tiep_theo(path[-1], path))
        path.append(path[0])

        total_length = np.sum(distance_matrix[path[:-1], path[1:]])
        all_paths.append(path)
        all_lengths.append(total_length)

        if total_length < best_distance:
            best_distance, best_path = total_length, path

    all_paths_history.append((all_paths, all_lengths))

    pheromone *= (1 - EVAPORATION)
    for path, length in zip(all_paths, all_lengths):
        pheromone[path[:-1], path[1:]] += Q / length
        pheromone[path[1:], path[:-1]] += Q / length

    print(f"Lần lặp {_ + 1}, Quãng đường tốt nhất: {best_distance:.2f}")

plt.figure(figsize=(10, 8))

for i in range(num_cities):
    plt.plot(cities[i][0], cities[i][1], 'ro', markersize=10)
    plt.text(cities[i][0] + 0.1, cities[i][1] + 0.1, str(i), fontsize=12)

for path in all_paths_history[-1][0]:
    plt.plot(cities[path, 0], cities[path, 1], 'gray', alpha=0.3, linewidth=1)

for i in range(len(best_path) - 1):
    a, b = best_path[i], best_path[i+1]
    pheromone_level = np.clip(pheromone[a, b] / np.max(pheromone), 0.1, 2)
    plt.plot(cities[[a, b], 0], cities[[a, b], 1], 'b-', linewidth=pheromone_level * 2, alpha=0.8)

plt.title(f"Tối ưu hóa đàn kiến: Quãng đường ngắn nhất = {best_distance:.2f}")
plt.grid(True)
plt.show()
