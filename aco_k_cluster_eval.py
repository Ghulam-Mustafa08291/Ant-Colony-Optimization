import networkx as nx
import math
import random
import numpy as np
from sklearn.cluster import KMeans

# --- 1. SETUP & LOAD MAP ---
NUM_ANTS = 20
NUM_ITERATIONS = 100
ALPHA, BETA, RHO, Q = 1.0, 5.0, 0.1, 100.0  # Tuned Beta for safety priority
K_CLUSTERS = 3 

def load_graph_data(filename):
    print(f"Loading graph from {filename}...")
    try:
        G = nx.read_graphml(filename)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return None, None, None, None, None

    nodes_list = list(G.nodes())
    node_to_index = {node_id: idx for idx, node_id in enumerate(nodes_list)}
    edges_dict = {}
    danger_points = []

    for u, v, data in G.edges(data=True):
        distance = float(data.get('length', 10.0))
        safety = float(data.get('safety', 50.0))
        edges_dict[(u, v)] = {'dist': distance, 'safety': safety}
        
        # Collect Danger Points for K-Means (Safety Threshold < 91.1)
        if safety < 91.1:
            try:
                ux = float(G.nodes[u].get('x', 0))
                uy = float(G.nodes[u].get('y', 0))
                vx = float(G.nodes[v].get('x', 0))
                vy = float(G.nodes[v].get('y', 0))
                mid_x = (ux + vx) / 2
                mid_y = (uy + vy) / 2
                danger_points.append([mid_x, mid_y])
            except:
                pass 

    print(f"Graph loaded. Found {len(danger_points)} high-risk road segments.")
    return nodes_list, edges_dict, np.array(danger_points), G, node_to_index

# --- 2. K-MEANS HOTSPOT GENERATION ---
def generate_hotspots(danger_points):
    if len(danger_points) < K_CLUSTERS:
        print("Not enough data for K-Means. Skipping.")
        return []
    
    print(f"Running K-Means to find {K_CLUSTERS} Crime Hotspots...")
    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=0, n_init=10).fit(danger_points)
    return kmeans.cluster_centers_

def evaluate_path_safety(path, hotspots, G):
    """ Metric: Average Distance from Danger Centroids (Higher is Better) """
    if not path or hotspots is None or len(hotspots) == 0: return 0
    
    total_dist = 0
    count = 0
    
    for node in path:
        try:
            nx_val = float(G.nodes[node].get('x', 0))
            ny_val = float(G.nodes[node].get('y', 0))
            
            # Find distance to NEAREST hotspot
            min_dist = float('inf')
            for hx, hy in hotspots:
                dist = math.sqrt((nx_val - hx)**2 + (ny_val - hy)**2)
                if dist < min_dist:
                    min_dist = dist
            
            total_dist += min_dist
            count += 1
        except:
            pass
            
    if count == 0: return 0
    return total_dist / count

# --- 3. FULL ACO ALGORITHM (Updated with Learning) ---
def run_aco(start, end, nodes, edges, weights):
    # Initialize Pheromones
    pheromones = {e: 1.0 for e in edges}
    
    best_path = None
    best_cost = float('inf')

    for _ in range(NUM_ITERATIONS):
        iteration_paths = [] # Store paths for update phase

        for _ in range(NUM_ANTS):
            curr = start
            path = [curr]
            visited = {curr}
            
            while curr != end:
                neighbors = [v for u, v in edges if u == curr and v not in visited]
                if not neighbors: break 
                
                probs = []
                denom = 0
                
                for n in neighbors:
                    edge = (curr, n)
                    tau = pheromones[edge]
                    
                    # Heuristic Cost
                    d = edges[edge]['dist']
                    s = 100.0 - edges[edge]['safety']
                    cost = (weights['dist'] * d) + (weights['safety'] * s)
                    eta = 1.0 / (cost + 0.01)
                    
                    # Probability Formula
                    v = (tau**ALPHA) * (eta**BETA)
                    probs.append(v)
                    denom += v
                
                if denom == 0: break
                probs = [p/denom for p in probs]
                
                next_node = random.choices(neighbors, weights=probs)[0]
                path.append(next_node)
                visited.add(next_node)
                curr = next_node
            
            if curr == end:
                path_c = 0
                for i in range(len(path)-1):
                    e = (path[i], path[i+1])
                    d = edges[e]['dist']
                    s = 100.0 - edges[e]['safety']
                    path_c += (weights['dist'] * d) + (weights['safety'] * s)
                
                iteration_paths.append((path, path_c))
                
                if path_c < best_cost:
                    best_cost = path_c
                    best_path = path

        # --- GLOBAL PHEROMONE UPDATE ---
        # 1. Evaporation
        for e in pheromones:
            pheromones[e] *= (1.0 - RHO)

        # 2. Deposit (Learning)
        for path, cost in iteration_paths:
            for i in range(len(path)-1):
                edge = (path[i], path[i+1])
                deposit = Q / (cost + 1.0) 
                pheromones[edge] += deposit
                    
    return best_path

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # Ensure you are passing 5 values to unpack, matching the return of load_graph_data
    NODES, EDGES, DANGER_POINTS, G, NODE_IDX = load_graph_data("karachi_clifton_roads_with_safety.graphml")
    
    if NODES:
        HOTSPOTS = generate_hotspots(DANGER_POINTS)
        print(f"Identified Hotspot Centers: \n{HOTSPOTS}")

        # Use IDs that exist in your map
        start_id = "286706786"
        end_id = "194416718"
        
        # Safety check for IDs
        if start_id in NODE_IDX and end_id in NODE_IDX:
            start = NODES[NODE_IDX[start_id]]
            end = NODES[NODE_IDX[end_id]]
        else:
            print("Warning: IDs not found, using defaults.")
            start = NODES[0]
            end = NODES[-1]

        print(f"\nPlanning route from {start} to {end}...")

        # 1. Shortest Path (Baseline)
        print("\n1. Calculating Shortest Path (Distance Optimized)...")
        # Use simple weights for baseline
        path_short = run_aco(start, end, NODES, EDGES, {'dist': 1.0, 'safety': 0.0})
        
        # 2. Safe Path (Smart Ants)
        print("2. Calculating Safe Path (Balanced)...")
        # Use Tuned Weights for Safety
        path_safe = run_aco(start, end, NODES, EDGES, {'dist': 1.0, 'safety': 20.0})

        # Final Evaluation
        print("\n" + "="*40)
        print("FINAL EVALUATION REPORT (Cluster Avoidance)")
        print("="*40)
        
        if path_short:
            score_short = evaluate_path_safety(path_short, HOTSPOTS, G)
            print(f"Shortest Path Avg Distance from Danger: {score_short:.2f} meters")
        
        if path_safe:
            score_safe = evaluate_path_safety(path_safe, HOTSPOTS, G)
            print(f"Safe ACO Path Avg Distance from Danger: {score_safe:.2f} meters")
            
            if path_short:
                diff = score_safe - score_short
                if diff > 0:
                    print(f"\nSUCCESS: Safe path keeps users {diff:.2f} meters further away from hotspots!")
                else:
                    print(f"\nResult: Safe path is not better (Diff: {diff:.2f}).")