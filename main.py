import random
import numpy as np
import heapq
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y, z, terrain_type='normal'):
        self.position = (x, y, z)
        self.visits = 0
        self.connections = []
        self.terrain_type = terrain_type
        self.temperature = random.uniform(20, 35)  # in Celsius

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

    def get_weighted_random_node(self, dragon):
        nodes = list(self.nodes.values())
        weights = []
        for node in nodes:
            weight = (node.visits + 1) * self.terrain_weight(node, dragon)
            weights.append(weight)
        return random.choices(nodes, weights=weights)[0]

    def terrain_weight(self, node, dragon):
        if node.terrain_type == 'basking' and dragon.needs_basking():
            return 5
        elif node.terrain_type == 'water' and dragon.is_thirsty():
            return 4
        elif node.terrain_type == 'hiding' and dragon.is_tired():
            return 3
        return 1

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

class KomodoDragon:
    def __init__(self, grid, start_node):
        self.grid = grid
        self.current_node = start_node
        self.energy = 100
        self.last_basking_time = 0
        self.last_drinking_time = 0

    def move(self, current_time):
        self.energy -= 1
        next_node = self.grid.get_weighted_random_node(self)
        path = self.grid.dijkstra(self.current_node, next_node)
        for index, node in enumerate(path):
            node.visits += (index + 1) / len(path)
            if node.terrain_type == 'basking':
                self.last_basking_time = current_time
                self.energy = min(100, self.energy + 10)
            elif node.terrain_type == 'water':
                self.last_drinking_time = current_time
        self.current_node = next_node

    def needs_basking(self):
        return self.energy < 50

    def is_thirsty(self):
        return self.last_drinking_time > 100

    def is_tired(self):
        return self.energy < 30

def plot_heatmap(grid, edges, iteration, dragons):
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

    # Add plotting of dragon positions
    for dragon in dragons:
        ax.scatter(*dragon.current_node.position, color='green', s=200, marker='*')

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

terrain_types = {
    (0, 0, 0): 'basking',
    (2, 2, 2): 'water',
    (4, 4, 2): 'hiding',
    # Add more terrain types as needed
}

grid = Grid3D(node_positions)
for pos, terrain in terrain_types.items():
    if pos in grid.nodes:
        grid.nodes[pos].terrain_type = terrain

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

dragons = [KomodoDragon(grid, grid.get_node((0, 0, 0))) for _ in range(3)]

total_iterations = 10000
plot_interval = total_iterations // 10

for i in range(total_iterations):
    for dragon in dragons:
        dragon.move(i)
    if (i + 1) % plot_interval == 0:
        plot_heatmap(grid, edges, i + 1, dragons)
