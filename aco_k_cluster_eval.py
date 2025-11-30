import networkx as nx
import math
import random
import numpy as np
from sklearn.cluster import KMeans

# --- 1. SETUP & LOAD MAP ---
NUM_ANTS = 20
NUM_ITERATIONS = 100
ALPHA, BETA, RHO, Q = 1.0, 2.0, 0.1, 100
K_CLUSTERS = 3 # We will look for 3 major "Danger Zones"

def load_graph_data(filename):
    print(f"Loading graph from {filename}...")
    try:
        G = nx.read_graphml(filename)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return None, None, None, None

    nodes_list = list(G.nodes())
    node_to_index = {node_id: idx for idx, node_id in enumerate(nodes_list)}  # MAPS INDEX TO NODE NUM (so i can use same start and end node as dani when comparing our algos)

    edges_dict = {}
    
    # We also need a list of "Danger Points" for K-Means
    danger_points = []

    for u, v, data in G.edges(data=True):
        # Get attributes (defaults if missing)
        distance = float(data.get('length', 10.0))
        safety = float(data.get('safety', 5.0))
        
        # Store edge
        edges_dict[(u, v)] = {'dist': distance, 'safety': safety}
        
        # --- K-MEANS PREP ---
        # If this road is dangerous (Safety < 5), add its location to our list
        if safety < 91.1:
            # We need coordinates. G.nodes[u] usually has 'x' and 'y'
            try:
                # Note: GraphML attributes are strings, cast to float
                ux = float(G.nodes[u].get('x', 0))
                uy = float(G.nodes[u].get('y', 0))
                vx = float(G.nodes[v].get('x', 0))
                vy = float(G.nodes[v].get('y', 0))
                
                # Use the midpoint of the road as the data point
                mid_x = (ux + vx) / 2
                mid_y = (uy + vy) / 2
                danger_points.append([mid_x, mid_y])
            except:
                pass # Skip if no coordinates found

    print(f"Graph loaded. Found {len(danger_points)} high-risk road segments.")
    return nodes_list, edges_dict, np.array(danger_points), G,node_to_index

# --- 2. K-MEANS HOTSPOT GENERATION ---

def generate_hotspots(danger_points):
    if len(danger_points) < K_CLUSTERS:
        print("Not enough data for K-Means. Skipping.")
        return
    
    print(f"Running K-Means to find {K_CLUSTERS} Crime Hotspots...")
    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=0, n_init=10).fit(danger_points)
    centroids = kmeans.cluster_centers_
    return centroids

def evaluate_path_safety(path, hotspots, G):
    """
    Metric: Average Distance from Danger Centroids (Higher is Better)
    """
    if not path or hotspots is None or len(hotspots) == 0: return 0
    
    total_dist = 0
    count = 0
    
    for node in path:
        try:
            nx_val = float(G.nodes[node].get('x', 0))
            ny_val = float(G.nodes[node].get('y', 0))
            
            # Find distance to NEAREST hotspot
            min_dist_to_hotspot = float('inf')
            for hx, hy in hotspots:
                dist = math.sqrt((nx_val - hx)**2 + (ny_val - hy)**2)
                if dist < min_dist_to_hotspot:
                    min_dist_to_hotspot = dist
            
            total_dist += min_dist_to_hotspot
            count += 1
        except:
            pass
            
    if count == 0: return 0
    return total_dist / count

# --- 3. ACO ALGORITHM (Simplified for brevity) ---
# (This is the same logic as before, just compacted)

def run_aco(start, end, nodes, edges, weights):
    pheromones = {e: 1.0 for e in edges}
    best_path = None
    best_cost = float('inf')

    for _ in range(NUM_ITERATIONS):
        for _ in range(NUM_ANTS):
            curr = start
            path = [curr]
            visited = {curr}
            
            while curr!= end:
                # Find neighbors
                neighbors = [v for u, v in edges if u == curr and v not in visited]
                if not neighbors: break
                
                # Calculate Probabilities
                probs = []
                denom = 0
                for n in neighbors:
                    edge = (curr, n)
                    tau = pheromones[edge]
                    
                    # Heuristic Cost
                    dist = edges[edge]['dist']
                    saf = 100.0 - edges[edge]['safety']  # Changed 10.0 to 100.0
                    cost = (weights['dist'] * dist) + (weights['safety'] * saf)
                    eta = 1.0 / (cost + 0.01)
                    
                    v = (tau**ALPHA) * (eta**BETA)
                    probs.append(v)
                    denom += v
                
                if denom == 0: break
                probs = [p/denom for p in probs]
                
                # Choose Next
                next_node = random.choices(neighbors, weights=probs)[0]  # Extract first element
                path.append(next_node)
                visited.add(next_node)
                curr = next_node
            
            if curr == end:
                # Calculate Path Cost
                path_c = 0
                for i in range(len(path)-1):
                    e = (path[i], path[i+1])
                    d = edges[e]['dist']
                    s = 100.0 - edges[e]['safety']
                    path_c += (weights['dist'] * d) + (weights['safety'] * s)
                
                if path_c < best_cost:
                    best_cost = path_c
                    best_path = path
                    
    return best_path

# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    NODES, EDGES, DANGER_POINTS, G ,NODES_TO_INDEX= load_graph_data("karachi_clifton_roads_with_safety.graphml") # <--- YOUR FILE
    
    if NODES:
        # Generate Hotspots
        HOTSPOTS = generate_hotspots(DANGER_POINTS)
        print(f"Identified Hotspot Centers at coordinates: \n{HOTSPOTS}")

        start_node_id=str(286706786)
        end_node_id=str(194416718)

        start_idx = NODES_TO_INDEX[start_node_id]
        end_idx = NODES_TO_INDEX[end_node_id]

        start = NODES [start_idx]
        end = NODES[end_idx] if len(NODES) > 20 else NODES[-1]
        print(f"\nPlanning route from {start} to {end}...")

        # --- SCENARIO 1: The "Shortest" Path (Dijkstra behavior) ---
        print("\n1. Calculating Shortest Path (Distance Optimized)...")
        path_short = run_aco(start, end, NODES, EDGES, {'dist': 1.0, 'safety': 0.0})
        
        # --- SCENARIO 2: The "Safe" Path (Your Algorithm) ---
        print("2. Calculating Safe Path (Balanced)...")
        path_safe = run_aco(start, end, NODES, EDGES, {'dist': 0.5, 'safety': 0.5})

        # --- FINAL EVALUATION ---
        print("\n" + "="*40)
        print("FINAL EVALUATION REPORT (K-Means)")
        print("="*40)
        
        # Check Shortest Path
        if path_short:
            score_short = evaluate_path_safety(path_short, HOTSPOTS, G)
            print(f"Shortest Path Safety Score: {score_short:.2f} meters")
        else:
            print("Shortest Path: NO ROUTE FOUND")

        # Check Safe Path
        if path_safe:
            score_safe = evaluate_path_safety(path_safe, HOTSPOTS, G)
            print(f"Safe ACO Path Safety Score: {score_safe:.2f} meters")
            
            # Only compare if both exist
            if path_short:
                diff = score_safe - score_short
                if diff > 0:
                    print(f"\nSUCCESS: Safe path is {diff:.2f} meters further from danger!")
                else:
                    print(f"\nResult: Safe path is not better (Diff: {diff:.2f}).")
        else:
            print("Safe ACO Path: NO ROUTE FOUND (Ants refused to take the risk!)")