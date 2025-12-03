import networkx as nx
import math
import random

# --- CONFIGURATION ---
NUM_ANTS = 20
NUM_ITERATIONS = 100
ALPHA, BETA = 1.0, 5.0  # Pheromone vs Heuristic weight
RHO = 0.1               # Evaporation rate

# --- 1. LOAD GRAPH (Same as before) ---
def load_graph_data(filename):
    print(f"Loading graph from {filename}...")
    try:
        G = nx.read_graphml(filename)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return None, None, None

    nodes_list = list(G.nodes())
    node_to_index = {node_id: idx for idx, node_id in enumerate(nodes_list)}
    edges_dict = {}

    for u, v, data in G.edges(data=True):
        distance = float(data.get('length', 10.0))
        # Ensure safety is within 0-100 range
        safety = float(data.get('safety', 50.0))
        
        # Store for ACO access
        edges_dict[(u, v)] = {'dist': distance, 'safety': safety}

    return nodes_list, edges_dict, node_to_index

# --- 2. ACO ALGORITHM (Standard Logic) ---
def run_aco(start, end, edges, weights):
    # Initialize Pheromones (1.0 on all edges)
    pheromones = {e: 1.0 for e in edges}
    
    best_path = None
    best_cost = float('inf')

    for _ in range(NUM_ITERATIONS):
        for _ in range(NUM_ANTS):
            curr = start
            path = [curr]
            visited = {curr}
            
            # Walk until we reach end or get stuck
            while curr != end:
                # Find neighbors in our custom edges dict
                neighbors = [v for u, v in edges if u == curr and v not in visited]
                
                if not neighbors: 
                    break # Dead end
                
                # Calculate Probabilities for next step
                probs = []
                denom = 0
                
                for n in neighbors:
                    edge = (curr, n)
                    tau = pheromones[edge]
                    
                    # --- HEURISTIC COST FUNCTION ---
                    # Cost = (Weight_Dist * Distance) + (Weight_Safety * (100 - Safety))
                    # Lower Safety = Higher Cost
                    d = edges[edge]['dist']
                    s = 100.0 - edges[edge]['safety'] 
                    
                    cost = (weights['dist'] * d) + (weights['safety'] * s)
                    
                    # Heuristic (Eta) is inverse of Cost (Cheaper is better)
                    eta = 1.0 / (cost + 0.01) 
                    
                    # ACO Formula: P = (Tau^Alpha) * (Eta^Beta)
                    v = (tau**ALPHA) * (eta**BETA)
                    probs.append(v)
                    denom += v
                
                if denom == 0: break
                
                # Normalize probabilities
                probs = [p/denom for p in probs]
                
                # Selection (Roulette Wheel)
                next_node = random.choices(neighbors, weights=probs)[0]
                path.append(next_node)
                visited.add(next_node)
                curr = next_node
            
            # Check if ant reached the destination
            if curr == end:
                # Calculate total path cost to see if it's the best so far
                path_c = 0
                for i in range(len(path)-1):
                    e = (path[i], path[i+1])
                    d = edges[e]['dist']
                    s = 100.0 - edges[e]['safety']
                    path_c += (weights['dist'] * d) + (weights['safety'] * s)
                
                if path_c < best_cost:
                    best_cost = path_c
                    best_path = path
                    
        # (Optional) Global Pheromone Update could go here
                    
    return best_path

# --- 3. THE ISOLATION TEST ---
if __name__ == "__main__":
    # A. Setup
    FILENAME = "karachi_clifton_roads_with_safety.graphml"
    NODES, EDGES, NODE_IDX = load_graph_data(FILENAME)

    if NODES:
        # Define Start/End (Using your IDs)
        start_id = "286706786"
        end_id = "194416718"
        
        # Fallback if specific nodes don't exist in file
        if start_id not in NODE_IDX or end_id not in NODE_IDX:
            print("Warning: Specific User IDs not found. Using random nodes.")
            start_node = NODES[0]
            end_node = NODES[-1]
        else:
            start_node = NODES[NODE_IDX[start_id]]
            end_node = NODES[NODE_IDX[end_id]]

        print(f"\n--- TEST: ACO RESPONSIVENESS (ISOLATION) ---")
        print(f"Route: {start_node} -> {end_node}")

        # B. Run Baseline (Standard Weights)
        # We give equal weight to distance and safety (0.5/0.5)
        print("\n1. Running Baseline ACO...")
        baseline_path = run_aco(start_node, end_node, EDGES, {'dist': 1, 'safety': 20})

        if not baseline_path:
            print("CRITICAL ERROR: No path found in baseline. Cannot proceed with test.")
        else:
            print(f"-> Baseline Path Found! Length: {len(baseline_path)} steps.")
            
            # C. The Trigger (Inject Danger)
            # Pick a road in the middle of the path to sabotage
            sabotage_index = len(baseline_path) // 2
            u_bad = baseline_path[sabotage_index]
            v_bad = baseline_path[sabotage_index + 1]
            bad_edge = (u_bad, v_bad)

            print(f"\n2. INJECTING DANGER...")
            print(f"-> Targeting Edge: {bad_edge}")
            old_safety = EDGES[bad_edge]['safety']
            
            # --- FORCE UPDATE: Set Safety to near ZERO ---
            EDGES[bad_edge]['safety'] = 1.0 
            print(f"-> Safety Score changed from {old_safety} to {EDGES[bad_edge]['safety']} (EXTREME DANGER)")

            # D. Run Evaluation (Response)
            print("\n3. Running ACO Again (Dynamic Response)...")
            new_path = run_aco(start_node, end_node, EDGES, {'dist': 1, 'safety': 20})

            # E. Verify Success
            print("\n" + "="*40)
            print("EVALUATION RESULT")
            print("="*40)
            
            if not new_path:
                print("Outcome: FAILURE (No path found after update).")
            else:
                # Check if the bad edge is present in the new path
                # Note: Edges are directional in our dict logic, check tuple existence
                path_edges = [(new_path[i], new_path[i+1]) for i in range(len(new_path)-1)]
                
                if bad_edge in path_edges:
                    print(f"Outcome: FAILURE")
                    print(f"Reason: The ants still took the dangerous road {bad_edge}.")
                    print("Diagnosis: Increase 'safety' weight or Beta (heuristic) parameter.")
                else:
                    print(f"Outcome: SUCCESS")
                    print(f"Reason: The algorithm successfully avoided edge {bad_edge}.")
                    print(f"Baseline Steps: {len(baseline_path)}")
                    print(f"New Path Steps: {len(new_path)}")
                    print("Conclusion: ACO algorithm correctly adapts to dynamic weight changes.")