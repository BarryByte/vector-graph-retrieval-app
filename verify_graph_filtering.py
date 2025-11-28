import requests
import sys

API_URL = "http://localhost:8000"

def test_graph_filtering():
    print("--- Testing TC-GRAPH-02: Relationship Filtering ---")
    
    # 1. Create Nodes A, B, C
    print("\n1. Creating Nodes A, B, C...")
    nodes = {}
    try:
        for name in ["NodeA", "NodeB", "NodeC"]:
            res = requests.post(f"{API_URL}/nodes", json={"title": name, "text": f"This is {name}", "metadata": {"type": "filter_test"}})
            if res.status_code != 200:
                print(f"Failed to create node {name}: {res.text}")
                sys.exit(1)
            nodes[name] = res.json()['id']
            print(f"Created {name}: {nodes[name]}")
            
        # 2. Create Edges with different types
        # A -> B (FRIEND)
        # B -> C (COLLEAGUE)
        print("\n2. Creating Edges...")
        edges = [
            (nodes["NodeA"], nodes["NodeB"], "FRIEND"),
            (nodes["NodeB"], nodes["NodeC"], "COLLEAGUE")
        ]
        
        for src, tgt, type_ in edges:
            res = requests.post(f"{API_URL}/edges", json={"source": src, "target": tgt, "type": type_, "weight": 1.0})
            if res.status_code != 200:
                print(f"Failed to create edge: {res.text}")
                sys.exit(1)
            
        # 3. Search from A with NO filter (Depth 2)
        # Should find A, B, C
        print("\n3. Search from NodeA (No Filter)...")
        res = requests.get(f"{API_URL}/search/graph", params={"start_id": nodes["NodeA"], "depth": 2})
        data = res.json()
        found_ids = {n['id'] for n in data.get("nodes", [])}
        print(f"Found: {len(found_ids)} nodes")
        if nodes["NodeC"] in found_ids:
            print("SUCCESS: Found NodeC without filter.")
        else:
            print("FAILURE: Did not find NodeC without filter.")
            sys.exit(1)
            
        # 4. Search from A with Filter ['FRIEND'] (Depth 2)
        # Should find A, B but NOT C (because B->C is COLLEAGUE)
        print("\n4. Search from NodeA (Filter=['FRIEND'])...")
        res = requests.get(f"{API_URL}/search/graph", params={"start_id": nodes["NodeA"], "depth": 2, "relationship_types": ["FRIEND"]})
        data = res.json()
        found_ids = {n['id'] for n in data.get("nodes", [])}
        print(f"Found: {len(found_ids)} nodes")
        
        if nodes["NodeB"] in found_ids and nodes["NodeC"] not in found_ids:
            print("SUCCESS: Found NodeB but NOT NodeC with 'FRIEND' filter.")
        else:
            print(f"FAILURE: Unexpected results. Found: {found_ids}")
            sys.exit(1)
            
        # 5. Search from A with Filter ['FRIEND', 'COLLEAGUE'] (Depth 2)
        # Should find A, B, C
        print("\n5. Search from NodeA (Filter=['FRIEND', 'COLLEAGUE'])...")
        res = requests.get(f"{API_URL}/search/graph", params={"start_id": nodes["NodeA"], "depth": 2, "relationship_types": ["FRIEND", "COLLEAGUE"]})
        data = res.json()
        found_ids = {n['id'] for n in data.get("nodes", [])}
        if nodes["NodeC"] in found_ids:
            print("SUCCESS: Found NodeC with both filters.")
        else:
            print("FAILURE: Did not find NodeC with both filters.")
            sys.exit(1)

    finally:
        # 6. Cleanup
        print("\n6. Cleanup...")
        for nid in nodes.values():
            requests.delete(f"{API_URL}/nodes/{nid}")
        print("Deleted nodes.")

if __name__ == "__main__":
    test_graph_filtering()
