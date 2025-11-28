import requests
import time
import sys

API_URL = "http://localhost:8000"

def test_cycle_handling():
    print("--- Testing TC-GRAPH-03: Cycle Handling ---")
    
    # 1. Create Nodes A, B, C
    print("\n1. Creating Nodes A, B, C...")
    nodes = {}
    try:
        for name in ["NodeA", "NodeB", "NodeC"]:
            res = requests.post(f"{API_URL}/nodes", json={"title": name, "text": f"This is {name}", "metadata": {"type": "cycle_test"}})
            if res.status_code != 200:
                print(f"Failed to create node {name}: {res.text}")
                sys.exit(1)
            nodes[name] = res.json()['id']
            print(f"Created {name}: {nodes[name]}")
            
        # 2. Create Cyclic Edges: A->B->C->A
        print("\n2. Creating Cyclic Edges (A->B->C->A)...")
        edges = [
            (nodes["NodeA"], nodes["NodeB"]),
            (nodes["NodeB"], nodes["NodeC"]),
            (nodes["NodeC"], nodes["NodeA"])
        ]
        
        for src, tgt in edges:
            res = requests.post(f"{API_URL}/edges", json={"source": src, "target": tgt, "type": "CYCLE_LINK", "weight": 1.0})
            if res.status_code != 200:
                print(f"Failed to create edge: {res.text}")
                sys.exit(1)
            
        # 3. Perform Graph Search from A with depth 3
        print("\n3. Performing Graph Search from NodeA (Depth 3)...")
        # Depth 3 is enough to traverse the full cycle A->B->C->A
        res = requests.get(f"{API_URL}/search/graph", params={"start_id": nodes["NodeA"], "depth": 3})
        
        if res.status_code != 200:
            print(f"Graph search failed: {res.text}")
            sys.exit(1)
            
        data = res.json()
        returned_nodes = data.get("nodes", [])
        returned_edges = data.get("edges", [])
        
        print(f"Returned Nodes: {len(returned_nodes)}")
        print(f"Returned Edges: {len(returned_edges)}")
        
        # Verify we got all 3 nodes
        returned_ids = {n['id'] for n in returned_nodes}
        if all(nid in returned_ids for nid in nodes.values()):
            print("SUCCESS: All nodes in cycle returned.")
        else:
            print("FAILURE: Missing nodes.")
            sys.exit(1)
            
        # Verify we didn't get stuck in infinite loop (response returned)
        print("SUCCESS: Graph search terminated successfully.")
        
    finally:
        # 4. Cleanup
        print("\n4. Cleanup...")
        for nid in nodes.values():
            requests.delete(f"{API_URL}/nodes/{nid}")
        print("Deleted nodes.")

if __name__ == "__main__":
    test_cycle_handling()
