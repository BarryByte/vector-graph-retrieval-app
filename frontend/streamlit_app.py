import streamlit as st
import requests
import json
from streamlit_agraph import agraph, Node, Edge, Config

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Hybrid Retrieval Demo", layout="wide")
st.title("🧠 Hybrid Vector + Graph Retrieval")

# Sidebar for Navigation
page = st.sidebar.selectbox("Choose a Mode", ["Ingestion", "Search", "Graph Visualization"])

if page == "Ingestion":
    st.header("📝 Document Ingestion")
    
    with st.form("ingest_form"):
        title = st.text_input("Document Title")
        text = st.text_area("Document Content", height=200)
        submitted = st.form_submit_button("Ingest Document")
        
        if submitted and text:
            with st.spinner("Ingesting..."):
                try:
                    res = requests.post(f"{API_URL}/nodes", json={
                        "title": title,
                        "text": text,
                        "metadata": {"source": "streamlit"}
                    })
                    if res.status_code == 200:
                        st.success(f"Document '{title}' ingested successfully!")
                        st.json(res.json())
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    st.markdown("---")
    st.header("🔗 Create Relationship")
    with st.form("edge_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            source = st.text_input("Source Node ID")
        with col2:
            target = st.text_input("Target Node ID")
        with col3:
            rel_type = st.text_input("Relationship Type", value="RELATED_TO")
            
        edge_submit = st.form_submit_button("Create Edge")
        
        if edge_submit and source and target:
            try:
                res = requests.post(f"{API_URL}/edges", json={
                    "source": source,
                    "target": target,
                    "type": rel_type,
                    "weight": 1.0
                })
                if res.status_code == 200:
                    st.success("Edge created!")
                    st.json(res.json())
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

elif page == "Search":
    st.header("🔍 Hybrid Search")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search Query")
    with col2:
        search_type = st.selectbox("Search Type", ["Hybrid", "Vector Only"])
    
    if search_type == "Hybrid":
        col_a, col_b = st.columns(2)
        with col_a:
            vector_weight = st.slider("Vector Weight (α)", 0.0, 1.0, 0.7)
        with col_b:
            graph_weight = st.slider("Graph Weight (β)", 0.0, 1.0, 0.3)
    
    if st.button("Search"):
        if query:
            endpoint = "/search/vector"
            payload = {"query_text": query, "top_k": 5}
            
            if search_type == "Hybrid":
                endpoint = "/search/hybrid"
                payload.update({
                    "vector_weight": vector_weight,
                    "graph_weight": graph_weight,
                    "graph_expand_depth": 1
                })
            
            with st.spinner("Searching..."):
                try:
                    res = requests.post(f"{API_URL}{endpoint}", json=payload)
                    if res.status_code == 200:
                        results = res.json()
                        if not results:
                            st.warning("No results found.")
                        else:
                            for item in results:
                                with st.expander(f"{item.get('score', 0):.4f} | {item.get('metadata', {}).get('title', 'Untitled')}"):
                                    st.markdown(f"**ID:** `{item['id']}`")
                                    st.write(item.get('text', ''))
                                    if 'graph_info' in item and item['graph_info']:
                                        st.info(f"Graph Info: {item['graph_info']}")
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

elif page == "Graph Visualization":
    st.header("🕸️ Graph Visualization")
    
    start_id = st.text_input("Start Node ID (for neighborhood)")
    
    if st.button("Visualize"):
        if start_id:
            with st.spinner("Fetching graph data..."):
                try:
                    res = requests.get(f"{API_URL}/search/graph", params={"start_id": start_id, "depth": 2})
                    if res.status_code == 200:
                        data = res.json()
                        
                        nodes = []
                        edges = []
                        added_nodes = set()
                        
                        # Add start node
                        nodes.append(Node(id=start_id, label="Start", color="#fb7e81", size=25))
                        added_nodes.add(start_id)
                        
                        for item in data:
                            node_data = item['node']
                            nid = node_data['id']
                            label = node_data.get('title') or node_data.get('name') or (node_data.get('text')[:10] + "..." if node_data.get('text') else "Node")
                            
                            if nid not in added_nodes:
                                nodes.append(Node(id=nid, label=label, size=15))
                                added_nodes.add(nid)
                            
                            # Since API doesn't return edges explicitly yet, we infer connections to start node or show disconnected
                            # Ideally, we update API to return edges. 
                            # For now, let's just show the nodes.
                            # Or we can try to fetch edges if we had an endpoint.
                        
                        config = Config(width=700, height=500, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
                        agraph(nodes=nodes, edges=edges, config=config)
                        
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
