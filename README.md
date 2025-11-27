# Hybrid Vector + Graph Retrieval System

A powerful retrieval system that combines **Semantic Vector Search** (FAISS) with **Graph Knowledge Traversal** (Neo4j) to provide context-aware search results.

## 🚀 Features

- **Hybrid Search**: Combines vector similarity scores with graph connectivity metrics.
- **Auto-Ingestion**:
  - **Text Chunking**: Automatically cleans and chunks text.
  - **Vector Embedding**: Uses `all-MiniLM-L6-v2` for dense retrieval.
  - **Auto-NER**: Uses Spacy to extract entities (Person, Org, GPE) and links them in the graph.
- **Graph Scoring**: Re-ranks results based on node centrality and relationship weights.
- **Interactive UI**: Streamlit dashboard for ingestion, search, and graph visualization.
- **Persistence**: Data is persisted in Neo4j (Docker volume) and FAISS (disk).

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.12
- **Database**: Neo4j (Graph), FAISS (Vector)
- **ML/NLP**: SentenceTransformers, Spacy
- **Frontend**: Streamlit, Streamlit-Agraph

## 📋 Prerequisites

- **Docker** & **Docker Compose** (for Neo4j)
- **Python 3.8+**

## ⚡ Quick Start

### 1. Start the Database
Start the Neo4j container using Docker Compose:
```bash
docker-compose up -d
```
*Neo4j will be available at http://localhost:7474 (User: `neo4j`, Pass: `password`)*

### 2. Setup Python Environment
Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the Spacy model for NER:
```bash
python -m spacy download en_core_web_sm
```

### 3. Run the Backend API
Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
*API Docs available at http://localhost:8000/docs*

### 4. Run the Frontend UI
In a new terminal (with `venv` activated), start the Streamlit app:
```bash
streamlit run frontend/streamlit_app.py
```
*The UI will open at http://localhost:8501*

## 🧪 Testing

Run the integration tests to verify the system:
```bash
pytest tests/test_integration.py
```

## 📂 Project Structure

```
.
├── app/
│   ├── main.py            # API Entrypoint
│   ├── models.py          # Pydantic Models
│   ├── database.py        # Neo4j & FAISS connection
│   └── services/
│       ├── ingestion.py   # Logic for Nodes, Edges, NER
│       └── search.py      # Vector, Graph, & Hybrid Search logic
├── frontend/
│   └── streamlit_app.py   # Interactive Dashboard
├── data/                  # Persisted data (Neo4j & FAISS)
├── tests/                 # Integration tests
├── docker-compose.yml     # Neo4j service config
└── requirements.txt       # Python dependencies
```
