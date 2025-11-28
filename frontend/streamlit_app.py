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
            alpha = st.slider("Vector Weight (α)", 0.0, 1.0, 0.7)
        with col_b:
            beta = 1.0 - alpha
            st.metric("Graph Weight (β)", f"{beta:.2f}")

        # Normalize so α + β = 1
        total = alpha + beta
        if total <= 0:
            alpha, beta = 1.0, 0.0   # default to pure vector if both zero
        else:
            alpha = alpha / total
            beta = beta/ total

        st.caption(f"Effective Weights → α = {alpha:.2f}, β = {beta:.2f} (normalized)")
    
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
                                        nodes.append(Node(id=nid, label=label, color=color, size=18))
                                    
                                    for edge_data in data.get("edges", []):
                                        edges.append(Edge(source=edge_data['source'], target=edge_data['target'], label=edge_data['type'], color="#CCCCCC"))
                                    
                                    config = Config(width=1000, height=800, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
                                    agraph(nodes=nodes, edges=edges, config=config)
                                    
                                    # Also list the nodes found
                                    # -------------------------------
                                    # Level-wise Indented "Found Nodes"
                                    # -------------------------------

                                    # Build adjacency list from edges
                                    # Build adjacency list from raw edge data (NOT Edge objects)
                                    adj = {}
                                    for e in data.get("edges", []):
                                        s = e.get("source")
                                        t = e.get("target")
                                        if s is None or t is None:
                                            continue
                                        adj.setdefault(s, []).append(t)
                                        adj.setdefault(t, []).append(s)

                                    # BFS from start_id to compute levels
                                    levels = {start_id: 0}
                                    queue = deque([start_id])

                                    while queue:
                                        current = queue.popleft()
                                        for nbr in adj.get(current, []):
                                            if nbr not in levels:
                                                levels[nbr] = levels[current] + 1
                                                queue.append(nbr)

                                    # Bucket Node objects by level
                                    level_buckets = {}
                                    for node in nodes:
                                        lvl = levels.get(node.id, -1)  # -1 if not connected in BFS
                                        level_buckets.setdefault(lvl, []).append(node)

                                    # Display formatted, level-wise list
                                    st.subheader("📌 Found Nodes (Level-wise)")

                                    for lvl in sorted(level_buckets.keys()):
                                        st.markdown(f"### Level {lvl}")
                                        for node in level_buckets[lvl]:
                                            indent = "&nbsp;" * (lvl * 8)  # HTML indentation
                                            st.markdown(
                                                f"{indent}- **{node.label}** (`{node.id}`)",
                                                unsafe_allow_html=True,
                                            )
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
                        "vector_weight": alpha,
                        "graph_weight": beta,
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
                                    # boost_badge = "🚀 Graph Boosted" if graph_info.get("expansion_bonus", 0) > 0 else ""
                                    
                                    with st.expander(f"{score:.4f} | {title}"):
                                        st.markdown(f"**ID:** `{item['id']}`")
                                        st.write(item.get('text', ''))
                                        
                                        if graph_info:
                                            st.markdown("#### 📊 Score Breakdown")
                                            c1, c2, c3, c4 = st.columns(4)
                                            c1.metric("Vector Score", f"{graph_info.get('vector_score_norm', 0):.2f}")
                                            c2.metric("Connectivity", f"{graph_info.get('connectivity_score_norm', 0):.2f}")
                                            c3.metric("Hops", f"{graph_info.get('hops', 0)}")
                                            # c4.metric("Expansion Bonus", f"{graph_info.get('expansion_bonus', 0)}")
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
    max_nodes = st.slider("Max nodes to display", 10, 300, 80, 10)
    max_neighbors_per_node = st.slider("Max neighbors per node", 2, 30, 10, 1)

    show_documents = st.checkbox("Show Documents", value=True)
    show_entities = st.checkbox("Show Entities", value=True)
    show_attributes = st.checkbox("Show Attributes / Values", value=False)  # off by default
    show_edge_labels_around_start = st.checkbox("Show edge labels only around start node", value=True)  

    if st.button("Visualize"):
        if start_id:
            with st.spinner("Fetching graph data..."):
                try:
                    res = requests.get(f"{API_URL}/search/graph", params={"start_id": start_id, "depth": 1})
                    if res.status_code == 200:
                        data = res.json()

                        raw_nodes = data.get("nodes", [])
                        raw_edges = data.get("edges", [])

                        # --- Build adjacency for BFS limiting ---
                        adjacency = {}
                        for e in raw_edges:
                            s = e["source"]
                            t = e["target"]
                            adjacency.setdefault(s, set()).add(t)
                            adjacency.setdefault(t, set()).add(s)

                        # --- BFS from start_id, capped by max_nodes & max_neighbors_per_node ---
                        selected_ids = set([start_id])
                        queue = [start_id]

                        while queue and len(selected_ids) < max_nodes:
                            current = queue.pop(0)
                            neighbors = list(adjacency.get(current, []))[:max_neighbors_per_node]
                            for nb in neighbors:
                                if len(selected_ids) >= max_nodes:
                                    break
                                if nb not in selected_ids:
                                    selected_ids.add(nb)
                                    queue.append(nb)

                        # --- Filter nodes by selected_ids + node-type checkboxes ---
                        def node_visible(node):
                            nid = node["id"]
                            if nid not in selected_ids:
                                return False

                            labels = node.get("labels", [])
                            ntype = node.get("type")

                            is_document = "Document" in labels or ntype == "Document"
                            is_entity = "Entity" in labels or ntype == "Entity"
                            is_attribute = "Attribute" in labels or "Value" in labels or ntype in ("Attribute", "Value")

                            if is_document and not show_documents:
                                return False
                            if is_entity and not show_entities:
                                return False
                            if is_attribute and not show_attributes:
                                return False

                            # default: keep
                            return True

                        filtered_nodes_data = [n for n in raw_nodes if node_visible(n)]
                        visible_ids = {n["id"] for n in filtered_nodes_data}

                        # --- Filter edges so both ends are visible & still within BFS set ---
                        filtered_edges_data = [
                            e for e in raw_edges
                            if e["source"] in visible_ids and e["target"] in visible_ids
                        ]

                        # --- Build Node objects (smaller size, shorter labels, type-based color) ---
                        nodes = []
                        for node_data in filtered_nodes_data:
                            nid = node_data["id"]
                            base_label = (
                                node_data.get("title")
                                or node_data.get("name")
                                or node_data.get("text", "Node")
                            )
                            # shorter label to avoid clutter
                            label = (base_label[:30] + "…") if len(base_label) > 30 else base_label

                            labels = node_data.get("labels", [])
                            ntype = node_data.get("type")

                            # default Document-blue
                            color = "#97C2FC"
                            if "Entity" in labels or ntype == "Entity":
                                color = "#FB7E81"  # red
                            if "Attribute" in labels or "Value" in labels or ntype in ("Attribute", "Value"):
                                color = "#9C9C9C"  # grey-ish for low-priority nodes

                            if nid == start_id:
                                color = "#FFFF00"  # yellow for start node

                            nodes.append(
                                Node(
                                    id=nid,
                                    label=label,
                                    color=color,
                                    size=15,  # slightly smaller to reduce overlap
                                )
                            )

                        # --- Build Edge objects (optionally label only edges touching start node) ---
                        edges = []
                        for edge_data in filtered_edges_data:
                            etype = edge_data["type"]
                            color = "#CCCCCC"
                            if etype == "MENTIONS":
                                color = "#FB7E81"
                            elif etype in ("RELATED_TO", "SEMANTIC_NEAR"):
                                color = "#97C2FC"

                            label = None
                            if not show_edge_labels_around_start:
                                label = etype
                            else:
                                if edge_data["source"] == start_id or edge_data["target"] == start_id:
                                    label = etype

                            edges.append(
                                Edge(
                                    source=edge_data["source"],
                                    target=edge_data["target"],
                                    label=label,
                                    color=color,
                                )
                            )

                        # --- Config: a bit more spacing & collapsible interaction ---
                        config = Config(
                            width=1100,
                            height=800,
                            directed=True,
                            nodeHighlightBehavior=True,
                            highlightColor="#F7A7A6",
                            collapsible=True,
                            # react-d3-graph config passthrough
                            node={"labelPosition": "top"},
                            link={"renderLabel": True},
                            d3={"gravity": -250, "linkLength": 140},
                        )

                        agraph(nodes=nodes, edges=edges, config=config)

                        st.info(f"Nodes shown: {len(nodes)} (from {len(raw_nodes)} total), "
                                f"Edges shown: {len(edges)} (from {len(raw_edges)} total)")
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
