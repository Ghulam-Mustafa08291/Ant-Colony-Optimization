import random
import math
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NUM_ANTS = 50
NUM_ITERATIONS = 100
ALPHA = 1.0  # importance of pheromone (memory)
BETA = 2.0   # importance of heuristic (sight)
RHO = 0.1    # pheromone evaporation rate
Q = 100      # pheromone deposit constant

# Global variables for graph data
NODES = []
EDGES = {}
PHEROMONES = {}
GRAPH = None  # Will hold the NetworkX graph

def load_graphml_network(graphml_file):
    """
    Load the Karachi road network from GraphML file.
    Extracts nodes and edges with their safety_index and length attributes.
    """
    global NODES, EDGES, PHEROMONES, GRAPH
    
    print(f"Loading graph from {graphml_file}...")
    GRAPH = nx.read_graphml(graphml_file)
    
    # Extract all node IDs
    NODES = list(GRAPH.nodes())
    print(f"Loaded {len(NODES)} nodes")
    
    # Extract all edges with their attributes
    EDGES = {}
    for u, v, data in GRAPH.edges(data=True):
        # Extract distance (length) - convert from string if needed
        dist = float(data.get('length', 100))  # default 100 if missing
        
        # Extract safety index (1-10 scale based on your GraphML)
        # If missing, use a neutral value of 5
        safety = int(data.get('safety_index', 5))
        
        EDGES[(u, v)] = {'dist': dist, 'safety': safety}
    
    print(f"Loaded {len(EDGES)} edges")
    
    # Initialize pheromone trails
    PHEROMONES = {edge: 1.0 for edge in EDGES}
    
    return GRAPH


def visualize_path(path, title, filename, color='red'):
    """
    Visualize the found path on the Karachi road network.
    
    Args:
        path: List of node IDs representing the path
        title: Title for the plot
        filename: Filename to save the visualization
        color: Color for the highlighted path
    """
    if GRAPH is None:
        print("Error: Graph not loaded")
        return
    
    if path is None or len(path) == 0:
        print(f"No path to visualize for {title}")
        return
    
    print(f"Generating visualization: {filename}...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get node positions from the graph (using x, y coordinates)
    pos = {}
    for node in GRAPH.nodes():
        # Extract x and y coordinates from node attributes
        node_data = GRAPH.nodes[node]
        x = float(node_data.get('x', 0))
        y = float(node_data.get('y', 0))
        pos[node] = (x, y)
    
    # Draw the entire network in light gray (background)
    nx.draw_networkx_edges(
        GRAPH, pos, 
        edge_color='lightgray', 
        width=0.5, 
        alpha=0.3,
        arrows=False,
        ax=ax
    )
    
    # Draw all nodes as small gray dots
    nx.draw_networkx_nodes(
        GRAPH, pos,
        node_size=10,
        node_color='lightgray',
        alpha=0.5,
        ax=ax
    )
    
    # Create edge list for the path
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    
    # Draw the path edges in the specified color (thicker and prominent)
    nx.draw_networkx_edges(
        GRAPH, pos,
        edgelist=path_edges,
        edge_color=color,
        width=3,
        alpha=0.9,
        arrows=True,
        arrowsize=15,
        arrowstyle='->',
        ax=ax
    )
    
    # Highlight start and end nodes
    start_node = path[0]
    end_node = path[-1]
    
    nx.draw_networkx_nodes(
        GRAPH, pos,
        nodelist=[start_node],
        node_size=200,
        node_color='green',
        node_shape='o',
        label='Start',
        ax=ax
    )
    
    nx.draw_networkx_nodes(
        GRAPH, pos,
        nodelist=[end_node],
        node_size=200,
        node_color='red',
        node_shape='s',
        label='End',
        ax=ax
    )
    
    # Highlight intermediate nodes in the path
    if len(path) > 2:
        nx.draw_networkx_nodes(
            GRAPH, pos,
            nodelist=path[1:-1],
            node_size=50,
            node_color=color,
            alpha=0.7,
            ax=ax
        )
    
    # Add title and labels
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Create legend
    green_patch = mpatches.Patch(color='green', label=f'Start Node')
    red_patch = mpatches.Patch(color='red', label=f'End Node')
    path_patch = mpatches.Patch(color=color, label=f'Path ({len(path)} nodes)')
    plt.legend(handles=[green_patch, red_patch, path_patch], loc='upper right', fontsize=10)
    
    # Remove axes for cleaner look
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    output_path = Path(__file__).parent / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)


def calculate_combined_cost(edge, weights):
    #find the "heurisits sight" (or its desirability ) for a single edge
    if edge not in EDGES:
        return 0.0 # No path, so 0 attractiveness

    dist_val = EDGES[edge]['dist']
    safety_score = EDGES[edge]['safety']  # Range: 1-100 (higher = safer)
    
    # Invert safety score to a cost (since higher safety = lower cost)
    # Safety ranges from 1-100, so we invert it: 100 - safety_score
    # So safety_score=100 (safest) -> safety_cost=0
    # And safety_score=1 (most dangerous) -> safety_cost=99
    safety_cost = 100 - safety_score

    # Calculate the weighted cost
    cost = (weights['dist'] * dist_val) + (weights['safety'] * safety_cost)
    
    # Heuristic (eta) is the inverse of cost (desirability)
    # Add a small value to avoid division by zero
    return 1.0 / (cost + 0.001)

def calculate_path_cost(path, weights):
    #this finds the fitness cost L(P) for ant's complete path to find reweard for best path 
    
    total_cost = 0
    for i in range(len(path) - 1):
        edge = (path[i], path[i+1])
        
        # This check is important if an ant gets stuck
        if edge not in EDGES:
            return float('inf') 

        # Here we sum the *actual* combined cost, not the heuristic
        dist_val = EDGES[edge]['dist']
        safety_score = EDGES[edge]['safety']  # Range: 1-100
        safety_cost = 100 - safety_score  # Invert: higher safety = lower cost
        cost = (weights['dist'] * dist_val) + (weights['safety'] * safety_cost)
        total_cost += cost
    return total_cost

def calculate_path_metrics(path):
    """
    Calculate separate distance and safety metrics for a path.
    Returns a dictionary with detailed metrics.
    """
    if path is None or len(path) < 2:
        return None
    
    total_distance = 0
    safety_scores = []
    
    for i in range(len(path) - 1):
        edge = (path[i], path[i+1])
        
        if edge not in EDGES:
            return None
        
        # Accumulate distance
        total_distance += EDGES[edge]['dist']
        
        # Collect safety scores
        safety_scores.append(EDGES[edge]['safety'])
    
    # Calculate safety statistics
    avg_safety = sum(safety_scores) / len(safety_scores) if safety_scores else 0
    min_safety = min(safety_scores) if safety_scores else 0
    max_safety = max(safety_scores) if safety_scores else 0
    
    return {
        'total_distance': total_distance,
        'avg_safety': avg_safety,
        'min_safety': min_safety,
        'max_safety': max_safety,
        'num_edges': len(safety_scores)
    }

def calculate_probabilities(current_node, visited_nodes, weights):
   
    probabilities = {}
    total_attractiveness = 0.0

    # Get actual neighbors from the graph (nodes that have an edge from current_node)
    possible_next_nodes = []
    if GRAPH is not None:
        # Use NetworkX to get successors (outgoing edges in directed graph)
        for neighbor in GRAPH.successors(current_node):
            if neighbor not in visited_nodes:
                possible_next_nodes.append(neighbor)
    else:
        # Fallback for simple test cases
        for node in NODES:
            if node not in visited_nodes and (current_node, node) in EDGES:
                possible_next_nodes.append(node)

    if not possible_next_nodes:
        return {} # Ant is at a dead end

    #now for  each possible next node, calculate its attractiveness
    for next_node in possible_next_nodes:
        edge = (current_node, next_node)
        
        # Get pheromone (tau) and heuristic (eta)
        pheromone_smell = PHEROMONES[edge]
        heuristic_sight = calculate_combined_cost(edge, weights) # Your logic!
        
        # This is the core formula from our discussion
        attractiveness = (pheromone_smell ** ALPHA) * (heuristic_sight ** BETA)
        
        probabilities[next_node] = attractiveness
        total_attractiveness += attractiveness

    # 3. Calculate final probabilities (normalize attractiveness)
    if total_attractiveness == 0:
        # If all paths are equally unattractive, pick one at random
        return {node: 1.0/len(possible_next_nodes) for node in possible_next_nodes}
        
    for node in probabilities:
        probabilities[node] = probabilities[node] / total_attractiveness
        
    return probabilities

def select_next_node(probabilities):
    """
    This is the "probabilistic" part. A TSP script will have this.
    It's like spinning a weighted roulette wheel.
    """
    r = random.random()
    cumulative_prob = 0
    for node, prob in probabilities.items():
        cumulative_prob += prob
        if r <= cumulative_prob:
            return node
    # Fallback in case of floating point rounding errors
    return list(probabilities.keys())[-1] 


# --- 4. The Main ACO Algorithm (from your pseudocode image) ---

def find_safe_path_aco(start_node, end_node, weights):
    """
    Main function to run the ACO simulation.
    """
    global PHEROMONES # We need to modify the global pheromones
    
    # Reset pheromones for this run
    PHEROMONES = {edge: 1.0 for edge in EDGES}
    
    global_best_path = None
    global_best_cost = float('inf')

    # 'While the stopping condition is not met do'
    for iteration in range(NUM_ITERATIONS):
        
        current_wave_paths = []  # FIXED: Initialize as empty list
        
        # 'Position each ant in a starting node'
        for ant in range(NUM_ANTS):
            
            current_path = [start_node]
            current_node = start_node
            visited_nodes = {start_node}

            # 'Repeat... Until every ant has built a solution'
            while current_node != end_node:
                
                # 'Choose the next node by applying the state transition rule'
                probs = calculate_probabilities(current_node, visited_nodes, weights)
                
                if not probs:
                    break # Ant got stuck in a dead end

                next_node = select_next_node(probs)

                # (Optional: 'Apply step-by-step pheromone update' - local update)
                # This is an advanced feature, can skip for the demo.

                # Move the ant
                current_path.append(next_node)
                visited_nodes.add(next_node)
                current_node = next_node
            
            # Add this ant's finished path to the wave's results
            if current_node == end_node:
                current_wave_paths.append(current_path)

        # --- End of Ant Wave ---
        
        # 'Update the best solution'
        wave_best_path = None
        wave_best_cost = float('inf')
        
        for path in current_wave_paths:
            cost = calculate_path_cost(path, weights)
            if cost < wave_best_cost:
                wave_best_cost = cost
                wave_best_path = path

        if wave_best_cost < global_best_cost:
            global_best_cost = wave_best_cost
            global_best_path = wave_best_path
            
        # 'Apply offline pheromone update'
        
        # 1. Evaporation
        for edge in PHEROMONES:
            PHEROMONES[edge] = PHEROMONES[edge] * (1.0 - RHO)
            
        # 2. Deposit (for ONLY the best path found so far)
        if global_best_path:
            # Check for divide-by-zero if cost is 0
            if global_best_cost == 0:
                deposit_amount = float('inf')
            else:
                deposit_amount = Q / global_best_cost
                
            for i in range(len(global_best_path) - 1):
                edge = (global_best_path[i], global_best_path[i+1])
                if edge in PHEROMONES: # Ensure edge exists
                    PHEROMONES[edge] = PHEROMONES[edge] + deposit_amount

    # 'End While'
    return global_best_path, global_best_cost


# --- 5. Main Demo ---

if __name__ == "__main__":
    # Load the Karachi road network
    graphml_file = Path(__file__).parent / "karachi_clifton_roads_with_safety.graphml"
    
    if not graphml_file.exists():
        print(f"Error: GraphML file not found at {graphml_file}")
        print("Please ensure 'karachi_clifton_roads_with_safety.graphml' is in the same directory.")
        exit(1)
    
    load_graphml_network(graphml_file)
    
    # Select start and end nodes from the loaded network
    # You can change these to any valid node IDs from your network
    if len(NODES) < 2:
        print("Error: Not enough nodes in the graph")
        exit(1)
    
    # Use first and last node as demo, or specify your own
    # For a real demo, you should specify actual node IDs you're interested in
    start = NODES[0]
    end = NODES[min(100, len(NODES)-1)]  # Pick a node that's not too far in the list
    # end = NODES[1]
    
    print(f"\n" + "="*60)
    print(f"Running ACO on Karachi Road Network")
    print(f"Start Node: {start}")
    print(f"End Node: {end}")
    print(f"="*60)
    
    # Run 1: "The Shortest Path" (Ignores safety)
    print("\n[1/3] Finding shortest path (prioritizing distance)...")
    weights_dist = {'dist': 1.0, 'safety': 0.0}
    path1, cost1 = find_safe_path_aco(start, end, weights_dist)
    print(f"\n--- Run 1: Shortest Path (dist=1.0, safety=0.0) ---")
    if path1:
        print(f"Path found with {len(path1)} nodes")
        print(f"Path: {' -> '.join(path1[:5])}{'...' if len(path1) > 5 else ''}")
        print(f"Weighted Cost: {cost1:.2f}")
        
        # Display detailed metrics
        metrics1 = calculate_path_metrics(path1)
        if metrics1:
            print(f"  📏 Total Distance: {metrics1['total_distance']:.2f} meters")
            print(f"  🛡️  Average Safety: {metrics1['avg_safety']:.2f}/100")
            print(f"  ⚠️  Min Safety: {metrics1['min_safety']}/100 | Max Safety: {metrics1['max_safety']}/100")
        
        visualize_path(path1, "ACO Path 1: Shortest Distance", "path1_shortest_distance.png", color='blue')
    else:
        print("No path found")

    # Run 2: "The Safest Path" (Ignores distance)
    print("\n[2/3] Finding safest path (prioritizing safety)...")
    weights_safe = {'dist': 0.0, 'safety': 1.0}
    path2, cost2 = find_safe_path_aco(start, end, weights_safe)
    print(f"\n--- Run 2: Safest Path (dist=0.0, safety=1.0) ---")
    if path2:
        print(f"Path found with {len(path2)} nodes")
        print(f"Path: {' -> '.join(path2[:5])}{'...' if len(path2) > 5 else ''}")
        print(f"Weighted Cost: {cost2:.2f}")
        
        # Display detailed metrics
        metrics2 = calculate_path_metrics(path2)
        if metrics2:
            print(f"  📏 Total Distance: {metrics2['total_distance']:.2f} meters")
            print(f"  🛡️  Average Safety: {metrics2['avg_safety']:.2f}/100")
            print(f"  ⚠️  Min Safety: {metrics2['min_safety']}/100 | Max Safety: {metrics2['max_safety']}/100")
        
        visualize_path(path2, "ACO Path 2: Safest Route", "path2_safest_route.png", color='green')
    else:
        print("No path found")
    
    # Run 3: "The 'Safe-Karachi' Path" (Balanced)
    print("\n[3/3] Finding balanced path (distance + safety)...")
    weights_balanced = {'dist': 0.5, 'safety': 0.5}
    path3, cost3 = find_safe_path_aco(start, end, weights_balanced)
    print(f"\n--- Run 3: Balanced 'Safe-Karachi' Path (dist=0.5, safety=0.5) ---")
    if path3:
        print(f"Path found with {len(path3)} nodes")
        print(f"Path: {' -> '.join(path3[:5])}{'...' if len(path3) > 5 else ''}")
        print(f"Weighted Cost: {cost3:.2f}")
        
        # Display detailed metrics
        metrics3 = calculate_path_metrics(path3)
        if metrics3:
            print(f"  📏 Total Distance: {metrics3['total_distance']:.2f} meters")
            print(f"  🛡️  Average Safety: {metrics3['avg_safety']:.2f}/100")
            print(f"  ⚠️  Min Safety: {metrics3['min_safety']}/100 | Max Safety: {metrics3['max_safety']}/100")
        
        visualize_path(path3, "ACO Path 3: Balanced (Distance + Safety)", "path3_balanced.png", color='orange')
    else:
        print("No path found")
    
    # Print comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    if path1 and path2 and path3 and metrics1 and metrics2 and metrics3:
        print(f"\n{'Metric':<25} {'Shortest':<15} {'Safest':<15} {'Balanced':<15}")
        print("-" * 70)
        print(f"{'Distance (m)':<25} {metrics1['total_distance']:<15.2f} {metrics2['total_distance']:<15.2f} {metrics3['total_distance']:<15.2f}")
        print(f"{'Avg Safety (/100)':<25} {metrics1['avg_safety']:<15.2f} {metrics2['avg_safety']:<15.2f} {metrics3['avg_safety']:<15.2f}")
        print(f"{'Min Safety (/100)':<25} {metrics1['min_safety']:<15} {metrics2['min_safety']:<15} {metrics3['min_safety']:<15}")
        print(f"{'Number of Nodes':<25} {len(path1):<15} {len(path2):<15} {len(path3):<15}")
        print(f"{'Weighted Cost':<25} {cost1:<15.2f} {cost2:<15.2f} {cost3:<15.2f}")
    
    print("\n" + "="*60)
    print("ACO Demo Complete!")
    print("Visualizations saved as PNG files in the current directory.")
    print("="*60)
