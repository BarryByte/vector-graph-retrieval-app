# File: frontend/streamlit_app.py
import streamlit as st
import requests
import json
from streamlit_agraph import agraph, Node, Edge, Config

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Hybrid Retrieval Demo", layout="wide")
st.title("🧠 Hybrid Vector + Graph Retrieval")

# Sidebar for Navigation
page = st.sidebar.selectbox("Choose a Mode", ["Ingestion", "Search", "Graph Visualization", "Database Inspector"])

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
        search_type = st.selectbox("Search Type", ["Hybrid", "Vector Only", "Graph Search"])
    
    if search_type == "Hybrid":
        col_a, col_b = st.columns(2)
        with col_a:
            vector_weight = st.slider("Vector Weight (α)", 0.0, 1.0, 0.7)
        with col_b:
            graph_weight = st.slider("Graph Weight (β)", 0.0, 1.0, 0.3)
    
    if st.button("Search"):
        if query:
            if search_type == "Graph Search":
                # Graph Search Logic (Text -> Vector -> ID -> Graph)
                with st.spinner("Resolving query and fetching graph..."):
                    try:
                        # 1. Resolve to ID
                        v_res = requests.post(f"{API_URL}/search/vector", json={"query_text": query, "top_k": 1})
                        if v_res.status_code == 200:
                            v_results = v_res.json()
                            if v_results:
                                start_id = v_results[0]['id']
                                st.info(f"Starting Graph Search from: {v_results[0].get('metadata', {}).get('title', 'Untitled')} (ID: {start_id})")
                                
                                # 2. Fetch Graph
                                g_res = requests.get(f"{API_URL}/search/graph", params={"start_id": start_id, "depth": 2})
                                if g_res.status_code == 200:
                                    data = g_res.json()
                                    
                                    # Render Graph
                                    nodes = []
                                    edges = []
                                    for node_data in data.get("nodes", []):
                                        nid = node_data['id']
                                        label = node_data.get('title') or node_data.get('name') or (node_data.get('text')[:15] + "..." if node_data.get('text') else "Node")
                                        color = "#FFFF00" if nid == start_id else ("#FB7E81" if "Entity" in node_data.get("labels", []) or node_data.get("type") else "#97C2FC")
                                        nodes.append(Node(id=nid, label=label, color=color, size=20))
                                    
                                    for edge_data in data.get("edges", []):
                                        edges.append(Edge(source=edge_data['source'], target=edge_data['target'], label=edge_data['type'], color="#CCCCCC"))
                                    
                                    config = Config(width=800, height=600, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
                                    agraph(nodes=nodes, edges=edges, config=config)
                                    
                                    # Also list the nodes found
                                    st.subheader("Found Nodes")
                                    for n in nodes:
                                        st.write(f"- **{n.label}** ({n.id})")
                                else:
                                    st.error(f"Graph Search Error: {g_res.text}")
                            else:
                                st.warning("No matching concepts found to start graph search.")
                        else:
                            st.error(f"Vector Resolution Error: {v_res.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

            else:
                # Existing Vector/Hybrid Search Logic
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
                                    score = item.get('score', 0)
                                    title = item.get('metadata', {}).get('title', 'Untitled')
                                    
                                    # Badge for Graph Boost
                                    graph_info = item.get('graph_info', {})
                                    boost_badge = "🚀 Graph Boosted" if graph_info.get("expansion_bonus", 0) > 0 else ""
                                    
                                    with st.expander(f"{score:.4f} | {title} {boost_badge}"):
                                        st.markdown(f"**ID:** `{item['id']}`")
                                        st.write(item.get('text', ''))
                                        
                                        if graph_info:
                                            st.markdown("#### 📊 Score Breakdown")
                                            c1, c2, c3, c4 = st.columns(4)
                                            c1.metric("Vector Score", f"{graph_info.get('vector_score_norm', 0):.2f}")
                                            c2.metric("Connectivity", f"{graph_info.get('connectivity_score_norm', 0):.2f}")
                                            c3.metric("Hops", f"{graph_info.get('hops', 0)}")
                                            c4.metric("Expansion Bonus", f"{graph_info.get('expansion_bonus', 0)}")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

elif page == "Graph Visualization":
    st.header("🕸️ Graph Visualization")
    
    search_mode = st.radio("Search Mode", ["By Node ID", "By Text Query"], horizontal=True)
    
    start_id = None
    
    if search_mode == "By Node ID":
        start_id = st.text_input("Start Node ID (for neighborhood)")
    else:
        text_query = st.text_input("Search Concept (e.g. 'Neo4j')")
        if text_query:
            # Resolve to ID via Vector Search
            try:
                v_res = requests.post(f"{API_URL}/search/vector", json={"query_text": text_query, "top_k": 1})
                if v_res.status_code == 200:
                    results = v_res.json()
                    if results:
                        start_id = results[0]['id']
                        st.info(f"Resolved '{text_query}' to Node ID: {start_id} ({results[0].get('metadata', {}).get('title', 'Untitled')})")
                    else:
                        st.warning("No matching concepts found.")
            except Exception as e:
                st.error(f"Resolution Error: {e}")

    if st.button("Visualize"):
        if start_id:
            with st.spinner("Fetching graph data..."):
                try:
                    res = requests.get(f"{API_URL}/search/graph", params={"start_id": start_id, "depth": 2})
                    if res.status_code == 200:
                        data = res.json()
                        
                        nodes = []
                        edges = []
                        
                        # Process Nodes
                        for node_data in data.get("nodes", []):
                            nid = node_data['id']
                            label = node_data.get('title') or node_data.get('name') or (node_data.get('text')[:15] + "..." if node_data.get('text') else "Node")
                            
                            # Color coding
                            color = "#97C2FC" # Default Blue (Document)
                            if "Entity" in node_data.get("labels", []) or node_data.get("type"): # Assuming we might pass labels or infer from props
                                color = "#FB7E81" # Red (Entity)
                            # Check if it's the start node
                            if nid == start_id:
                                color = "#FFFF00" # Yellow
                            
                            nodes.append(Node(id=nid, label=label, color=color, size=20))
                        
                        # Process Edges
                        for edge_data in data.get("edges", []):
                            # Color coding edges
                            color = "#CCCCCC"
                            if edge_data['type'] == "MENTIONS":
                                color = "#FB7E81" # Redish
                            elif edge_data['type'] == "RELATED_TO":
                                color = "#97C2FC" # Blueish
                                
                            edges.append(Edge(
                                source=edge_data['source'], 
                                target=edge_data['target'], 
                                label=edge_data['type'],
                                color=color
                            ))
                        
                        config = Config(width=800, height=600, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
                        agraph(nodes=nodes, edges=edges, config=config)
                        
                        st.info(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
                        
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

elif page == "Database Inspector":
    st.header("🕵️ Database Inspector")
    
    tab1, tab2, tab3 = st.tabs(["Neo4j Documents", "Neo4j Entities", "FAISS Index"])
    
    with tab1:
        st.subheader("Stored Documents")
        if st.button("Refresh Documents"):
            try:
                res = requests.get(f"{API_URL}/debug/documents")
                if res.status_code == 200:
                    docs = res.json()
                    st.write(f"Total Documents: {len(docs)}")
                    
                    # Convert to list of dicts for dataframe
                    doc_list = []
                    for d in docs:
                        doc_list.append({
                            "ID": d.get("id"),
                            "Title": d.get("title"),
                            "Vector ID": d.get("vector_id"),
                            "Text": d.get("text")[:50] + "..." if d.get("text") else ""
                        })
                    
                    st.dataframe(doc_list, use_container_width=True)
                    
                    # Detail View
                    st.markdown("### Document Details")
                    selected_id = st.selectbox("Select Document ID to inspect", [d["ID"] for d in doc_list])
                    if selected_id:
                        selected_doc = next((d for d in docs if d["id"] == selected_id), None)
                        st.json(selected_doc)
                        
                        if selected_doc.get("vector_id") is not None:
                            vid = selected_doc["vector_id"]
                            st.markdown(f"**Vector Embedding (ID: {vid})**")
                            v_res = requests.get(f"{API_URL}/debug/faiss/vector/{vid}")
                            if v_res.status_code == 200:
                                vec_data = v_res.json()["embedding"]
                                st.write(f"Dimension: {len(vec_data)}")
                                st.line_chart(vec_data)
                            else:
                                st.warning("Could not fetch vector data.")
                else:
                    st.error("Failed to fetch documents")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with tab2:
        st.subheader("Stored Entities")
        if st.button("Refresh Entities"):
            try:
                res = requests.get(f"{API_URL}/debug/entities")
                if res.status_code == 200:
                    ents = res.json()
                    st.write(f"Total Entities: {len(ents)}")
                    
                    ent_list = []
                    for e in ents:
                        ent_list.append({
                            "ID": e.get("id"),
                            "Name": e.get("name"),
                            "Type": e.get("type")
                        })
                    
                    st.dataframe(ent_list, use_container_width=True)
                else:
                    st.error("Failed to fetch entities")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab3:
        st.subheader("Vector Index Stats")
        if st.button("Refresh Stats"):
            try:
                res = requests.get(f"{API_URL}/debug/faiss/info")
                if res.status_code == 200:
                    info = res.json()
                    st.json(info)
                    
                    st.markdown("### ID Mapping")
                    st.write("Mapping from FAISS Vector ID to Neo4j Document ID:")
                    st.dataframe(info.get("id_map", {}), use_container_width=True)
                else:
                    st.error("Failed to fetch FAISS info")
            except Exception as e:
                st.error(f"Error: {e}")
