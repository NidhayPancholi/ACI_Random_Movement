import random
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Node:
    def __init__(self, x, y, z, terrain_type='normal'):
        self.position = (x, y, z)
        self.visits = 0
        self.connections = []
        self.terrain_type = terrain_type
        self.temperature = random.uniform(20, 35)  # in Celsius
        self.humidity = random.uniform(40, 80)     # in percentage
        self.comfort_score = 0  # Will be calculated based on temperature and humidity

    def calculate_comfort_score(self, dragon):
        # Optimal conditions for Komodo dragons
        optimal_temp = 28
        optimal_humidity = 70
        
        # Calculate deviation from optimal conditions
        temp_deviation = abs(self.temperature - optimal_temp)
        humidity_deviation = abs(self.humidity - optimal_humidity)
        
        # Convert deviations to a comfort score (higher is better)
        self.comfort_score = 100 - (temp_deviation * 5 + humidity_deviation * 0.5)
        
        # Adjust for terrain type and dragon needs
        if self.terrain_type == 'basking' and dragon.needs_basking():
            self.comfort_score *= 1.5
        elif self.terrain_type == 'water' and dragon.is_thirsty():
            self.comfort_score *= 2  # Increased attraction to water when thirsty
        elif self.terrain_type == 'hiding' and dragon.is_tired():
            self.comfort_score *= 2

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
        current_node = dragon.current_node
        nodes_list = list(self.nodes.values())  # Consider all nodes in the grid
        weights = []

        # Calculate comfort scores for all nodes
        for node in nodes_list:
            node.calculate_comfort_score(dragon)

        # Calculate weights with adjacent bonus
        for node in nodes_list:
            weight = node.comfort_score
            
            # Add adjacent bonus (1.1x) for nodes connected to current position
            if node in current_node.connections:
                weight *= 1.1  # 10% bonus for adjacent nodes
            
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]

        # Select next node
        next_node = random.choices(nodes_list, weights=weights)[0]
        
        # Increment visit count for the selected node
        next_node.visits += 1
        
        return next_node

    def print_decision_info(self, current_node, nodes_list, weights, chosen_node):
        print("\nDecision Making Information:")
        print(f"Current Position: {current_node.position}")
        print(f"Current Temperature: {current_node.temperature:.1f}°C")
        print(f"Current Humidity: {current_node.humidity:.1f}%")
        print("\nPossible Destinations (showing top 5 weighted nodes):")
        
        # Create list of nodes with their weights
        node_weights = list(zip(nodes_list, weights))
        # Sort by weight in descending order
        node_weights.sort(key=lambda x: x[1], reverse=True)
        
        # Print top 5 options
        for node, weight in node_weights[:1]:
            print(f"\nNode at {node.position}:")
            print(f"  Temperature: {node.temperature:.1f}°C")
            print(f"  Humidity: {node.humidity:.1f}%")
            print(f"  Terrain: {node.terrain_type}")
            print(f"  Comfort Score: {node.comfort_score:.1f}")
            print(f"  Final Weight: {weight:.3f}")
            if node in current_node.connections:
                print("  ** ADJACENT NODE **")
            if node == chosen_node:
                print("  ** CHOSEN DESTINATION **")
        
        print(f"\nChosen Node Position: {chosen_node.position}")

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
        self.hydration = 100
        self.last_basking_time = 0
        self.last_drinking_time = 0
        self.basking_duration = 0  # Track how long it's been basking
        self.is_basking = False    # Flag to track if currently basking
        self.preferred_temp_range = (25, 35)
        self.preferred_humidity_range = (60, 80)

    def move(self, current_time):
        # Check if currently basking and should continue
        if self.is_basking and self.basking_duration < self.get_desired_basking_time():
            self.basking_duration += 1
            self.energy = min(100, self.energy + 2 * self.basking_duration)  # Energy gain increases with basking time
            return  # Stay in the same spot
        
        # Reset basking status if we're moving
        self.is_basking = False
        self.basking_duration = 0
        
        # Regular movement logic
        self.energy = max(0, self.energy - 1)
        self.hydration = max(0, self.hydration - 2)
        
        next_node = self.grid.get_weighted_random_node(self)
        
        # Update status based on destination terrain
        if next_node.terrain_type == 'basking' and self.needs_basking():
            self.last_basking_time = current_time
            self.is_basking = True
            self.basking_duration = 1
            self.energy = min(100, self.energy + 2)  # Initial energy gain
        elif next_node.terrain_type == 'water':
            self.last_drinking_time = current_time
            self.hydration = min(100, self.hydration + 30)
            
        self.current_node = next_node

    def get_desired_basking_time(self):
        # Determine how long the dragon should bask based on energy level
        if self.energy < 30:
            return 10  # Stay longer if energy is very low
        elif self.energy < 50:
            return 7   # Stay medium duration if energy is moderately low
        else:
            return 5   # Stay shorter time if energy is higher

    def needs_basking(self):
        return self.energy < 50

    def is_thirsty(self):
        return self.hydration < 50

    def is_tired(self):
        return self.energy < 30

def plot_heatmap(grid, edges, iteration, dragon):
    fig = plt.figure(figsize=(15, 10))
    
    # Create main 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot nodes
    x = [node.position[0] for node in grid.nodes.values()]
    y = [node.position[1] for node in grid.nodes.values()]
    z = [node.position[2] for node in grid.nodes.values()]
    visits = [node.visits for node in grid.nodes.values()]
    
    # Plot nodes with visit frequency colors
    sc = ax1.scatter(x, y, z, c=visits, cmap='YlOrRd', marker='o', s=100)
    plt.colorbar(sc, ax=ax1, label='Number of Visits')

    # Plot edges
    for edge in edges:
        pos1, pos2 = edge
        x_vals = [pos1[0], pos2[0]]
        y_vals = [pos1[1], pos2[1]]
        z_vals = [pos1[2], pos2[2]]
        ax1.plot(x_vals, y_vals, z_vals, color='gray', alpha=0.5)

    # Plot special terrain types
    for node in grid.nodes.values():
        if node.terrain_type == 'basking':
            ax1.scatter(*node.position, color='red', s=200, marker='^', 
                       label='Basking Spot' if 'Basking Spot' not in ax1.get_legend_handles_labels()[1] else "")
        elif node.terrain_type == 'water':
            ax1.scatter(*node.position, color='blue', s=200, marker='s', 
                       label='Water' if 'Water' not in ax1.get_legend_handles_labels()[1] else "")
        elif node.terrain_type == 'hiding':
            ax1.scatter(*node.position, color='brown', s=200, marker='v', 
                       label='Hiding Spot' if 'Hiding Spot' not in ax1.get_legend_handles_labels()[1] else "")

    # Plot dragon's current position
    ax1.scatter(*dragon.current_node.position, color='green', s=200, marker='*', label='Komodo Dragon')

    # Add labels and title
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Visit Frequency Heatmap (Iteration {iteration})')
    ax1.legend()

    # Create second subplot for information
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    # Add text information
    info_text = (
        f"Iteration: {iteration}\n\n"
        f"Dragon Status:\n"
        f"Energy Level: {dragon.energy}\n"
        f"Hydration Level: {dragon.hydration}\n"
        f"Last Basking: {iteration - dragon.last_basking_time} steps ago\n"
        f"Last Drinking: {iteration - dragon.last_drinking_time} steps ago\n"
        f"Currently Basking: {'Yes' if dragon.is_basking else 'No'}\n"
        f"Basking Duration: {dragon.basking_duration if dragon.is_basking else 0} steps\n\n"
        f"Current Node Info:\n"
        f"Position: {dragon.current_node.position}\n"
        f"Temperature: {dragon.current_node.temperature:.1f}°C\n"
        f"Humidity: {dragon.current_node.humidity:.1f}%\n"
        f"Terrain Type: {dragon.current_node.terrain_type}\n"
        f"Comfort Score: {dragon.current_node.comfort_score:.1f}\n"
        f"Visits: {dragon.current_node.visits}\n\n"
        f"Environment Stats:\n"
        f"Most Visited Node: {max(grid.nodes.values(), key=lambda x: x.visits).position}\n"
        f"Max Visits: {max(node.visits for node in grid.nodes.values())}\n"
        f"Total Visits: {sum(node.visits for node in grid.nodes.values())}\n\n"
        f"Temperature Range: {min(node.temperature for node in grid.nodes.values()):.1f}°C - "
        f"{max(node.temperature for node in grid.nodes.values()):.1f}°C\n"
        f"Humidity Range: {min(node.humidity for node in grid.nodes.values()):.1f}% - "
        f"{max(node.humidity for node in grid.nodes.values()):.1f}%"
    )
    
    ax2.text(0, 0.5, info_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
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
    (1, 4, 2): 'water',  # Add additional water location
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

# Update the main simulation loop
total_iterations = 1000
plot_interval = 100

# Initialize single dragon
dragon = KomodoDragon(grid, grid.get_node((0, 0, 0)))

for i in range(total_iterations):
    dragon.move(i)
    if (i + 1) % plot_interval == 0:
        plot_heatmap(grid, edges, i + 1, dragon)
