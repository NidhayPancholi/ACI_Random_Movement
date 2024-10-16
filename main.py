import random
import numpy as np
import heapq
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y, z):
        self.position = (x, y, z)
        self.visits = 0
        self.connections = []

    def add_connection(self, node):
        self.connections.append(node)

    def __lt__(self, other):
        return self.position < other.position

class Grid3D:
    def __init__(self, node_positions):
        self.nodes = {pos: Node(*pos) for pos in node_positions}

    def connect_nodes(self, pos1, pos2):
        node1 = self.nodes[pos1]
        node2 = self.nodes[pos2]
        node1.add_connection(node2)
        node2.add_connection(node1)

    def get_node(self, position):
        return self.nodes.get(position)

    def get_weighted_random_node(self):
        k = 3
        nodes = list(self.nodes.values())
        total_visits = sum(node.visits for node in nodes)
        if total_visits == 0:
            return random.choice(nodes)
        probabilities = [(node.visits + k) / (total_visits + len(nodes) * k) for node in nodes]
        return np.random.choice(nodes, p=probabilities)

    def dijkstra(self, start, goal):
        queue = [(0, start)]
        distances = {node: float('inf') for node in self.nodes.values()}
        distances[start] = 0
        previous_nodes = {node: None for node in self.nodes.values()}

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            if current_node == goal:
                path = []
                while previous_nodes[current_node]:
                    path.append(current_node)
                    current_node = previous_nodes[current_node]
                path.append(start)
                return path[::-1]

            for neighbor in current_node.connections:
                distance = current_distance + 1  # Assuming all edges have equal weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(queue, (distance, neighbor))

        return []

class Animal:
    def __init__(self, grid, start_node):
        self.grid = grid
        self.current_node = start_node

    def move(self):
        next_node = self.grid.get_weighted_random_node()
        path = self.grid.dijkstra(self.current_node, next_node)
        for index,node in enumerate(path):
            #print(f"Animal moved to {node.position}")
            node.visits += index/len(path)
        self.current_node = next_node


def plot_heatmap(grid, edges, iteration):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [node.position[0] for node in grid.nodes.values()]
    y = [node.position[1] for node in grid.nodes.values()]
    z = [node.position[2] for node in grid.nodes.values()]
    visits = [node.visits for node in grid.nodes.values()]

    # Increase the size of the nodes
    sc = ax.scatter(x, y, z, c=visits, cmap='hot', marker='o', s=100)
    plt.colorbar(sc, ax=ax, label='Number of Visits')

    # Plotting the edges
    for edge in edges:
        pos1, pos2 = edge
        x_vals = [pos1[0], pos2[0]]
        y_vals = [pos1[1], pos2[1]]
        z_vals = [pos1[2], pos2[2]]
        ax.plot(x_vals, y_vals, z_vals, color='gray')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Heatmap of Node Visits with Edges (Iteration {iteration})')

    plt.show()


# Example usage
node_positions = [
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 2), (1, 1, 2),
    (1, 1, 1), (1, 1, 0), (1, 0, 0), (1, 0, 1), (2, 0, 1),
    (2, 1, 1), (2, 1, 2), (3, 1, 2), (3, 2, 2), (4, 2, 2),
    (4, 3, 2), (4, 4, 2), (3, 4, 2), (2, 4, 2), (1, 4, 2),
    (0, 4, 2), (0, 3, 2), (0, 2, 2), (0, 1, 2)
]

grid = Grid3D(node_positions)

# Ensure the graph is connected by adding edges
edges = [
    ((0, 0, 0), (0, 0, 1)), ((0, 0, 1), (0, 0, 2)), ((0, 0, 2), (0, 1, 2)),
    ((0, 1, 2), (1, 1, 2)), ((1, 1, 2), (1, 1, 1)), ((1, 1, 1), (1, 1, 0)),
    ((1, 1, 0), (1, 0, 0)), ((1, 0, 0), (1, 0, 1)), ((1, 0, 1), (2, 0, 1)),
    ((2, 0, 1), (2, 1, 1)), ((2, 1, 1), (2, 1, 2)), ((2, 1, 2), (3, 1, 2)),
    ((3, 1, 2), (3, 2, 2)), ((3, 2, 2), (4, 2, 2)), ((4, 2, 2), (4, 3, 2)),
    ((4, 3, 2), (4, 4, 2)), ((4, 4, 2), (3, 4, 2)), ((3, 4, 2), (2, 4, 2)),
    ((2, 4, 2), (1, 4, 2)), ((1, 4, 2), (0, 4, 2)), ((0, 4, 2), (0, 3, 2)),
    ((0, 3, 2), (0, 2, 2)), ((0, 2, 2), (0, 1, 2))
]

for edge in edges:
    grid.connect_nodes(*edge)

animal = Animal(grid, grid.get_node((0, 0, 0)))

total_iterations = 10000
plot_interval = total_iterations // 10  # 20% intervals

for i in range(total_iterations):
    animal.move()
    if (i + 1) % plot_interval == 0:
        plot_heatmap(grid, edges, i + 1)

