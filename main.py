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
    def __init__(self, size_x, size_y, size_z):
        self.nodes = {}
        for x in range(size_x):
            for y in range(size_y):
                for z in range(size_z):
                    self.nodes[(x, y, z)] = Node(x, y, z)

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
        weights = [1 / (node.visits + k) for node in nodes]
        probabilities = [weight / sum(weights) for weight in weights]
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
        for node in path:
            print(f"Animal moved to {node.position}")
            node.visits += 1
        self.current_node = next_node

grid = Grid3D(5, 5, 5)
grid.connect_nodes((0, 0, 0), (0, 0, 1))
grid.connect_nodes((0, 0, 1), (0, 0, 2))
grid.connect_nodes((0, 0, 2), (0, 1, 2))
grid.connect_nodes((0, 1, 2), (1, 1, 2))
grid.connect_nodes((1, 1, 2), (1, 1, 1))
grid.connect_nodes((1, 1, 1), (1, 1, 0))
grid.connect_nodes((1, 1, 0), (1, 0, 0))
grid.connect_nodes((1, 0, 0), (1, 0, 1))

# Add more connections
grid.connect_nodes((1, 0, 1), (2, 0, 1))
grid.connect_nodes((2, 0, 1), (2, 1, 1))
grid.connect_nodes((2, 1, 1), (2, 1, 2))
grid.connect_nodes((2, 1, 2), (3, 1, 2))
grid.connect_nodes((3, 1, 2), (3, 2, 2))
grid.connect_nodes((3, 2, 2), (4, 2, 2))
grid.connect_nodes((4, 2, 2), (4, 3, 2))
grid.connect_nodes((4, 3, 2), (4, 4, 2))
grid.connect_nodes((4, 4, 2), (3, 4, 2))
grid.connect_nodes((3, 4, 2), (2, 4, 2))
grid.connect_nodes((2, 4, 2), (1, 4, 2))
grid.connect_nodes((1, 4, 2), (0, 4, 2))
grid.connect_nodes((0, 4, 2), (0, 3, 2))
grid.connect_nodes((0, 3, 2), (0, 2, 2))
grid.connect_nodes((0, 2, 2), (0, 1, 2))

animal = Animal(grid, grid.get_node((0, 0, 0)))

for _ in range(10000):
    animal.move()



# Plotting the 3D heatmap
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [node.position[0] for node in grid.nodes.values()]
y = [node.position[1] for node in grid.nodes.values()]
z = [node.position[2] for node in grid.nodes.values()]
visits = [node.visits for node in grid.nodes.values()]

sc = ax.scatter(x, y, z, c=visits, cmap='hot', marker='o')
plt.colorbar(sc, ax=ax, label='Number of Visits')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Heatmap of Node Visits')

plt.show()