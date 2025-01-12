import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from neo4j import GraphDatabase
from pyvis.network import Network
import requests
import json
import random

# --- Configuration ---
neo4j_uri = st.secrets["NEO4J_URI"]
neo4j_user = st.secrets["NEO4J_USER"]
neo4j_password = st.secrets["NEO4J_PASSWORD"]

modus_api_url = st.secrets.get("MODUS_API_URL", None)  # Ensure it works if not provided
modus_api_key = st.secrets.get("MODUS_API_KEY", None)  # Ensure it works if not provided

# Data file paths
file1_path = 'data/cancer_predisposition_variants.csv'
file2_path = 'data/gene_list.csv'

# --- Load Data ---
file1 = pd.read_csv(file1_path)
file2 = pd.read_csv(file2_path)

# List of prostate cancer-associated genes
prostate_genes = [
    "BRCA1", "BRCA2", "HOXB13", "ATM", "CHEK2",
    "MSH2", "MSH6", "MLH1", "PMS2", "NBN",
    "PALB2", "RNASEL"
]

# Filter rows in both files for these genes
file1_prostate = file1[file1['symbol'].isin(prostate_genes)]
file2_prostate = file2[
    (file2['hgnc_symbol'].isin(prostate_genes)) |
    (file2['cosmic_gene_symbol'].isin(prostate_genes))
]

# Merge datasets
integrated_data = pd.merge(file1_prostate, file2_prostate, on='gene_id', suffixes=('_variants', '_annotations'))
if 'chr_stop' not in integrated_data.columns:
    integrated_data['chr_stop'] = None

# --- Feature Engineering ---
mutation_counts = integrated_data.groupby(['symbol', 'effect']).size().unstack(fill_value=0)
mutation_counts['total_mutations'] = mutation_counts.sum(axis=1)
data_ml = mutation_counts.reset_index()

X = data_ml.drop(columns=['symbol'])
y = (data_ml['total_mutations'] > data_ml['total_mutations'].median()).astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Modus API Integration ---
def analyze_with_modus(features, labels):
    """Analyze data with Modus API."""
    try:
        response = requests.post(
            modus_api_url,
            json={"features": features.tolist(), "labels": labels.tolist()},
            headers={"Authorization": f"Bearer {modus_api_key}"},
            timeout=30  # Add a timeout to avoid hanging
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Modus API unavailable: {e}")
        return None

modus_results = None
if modus_api_url and modus_api_key:
    modus_results = analyze_with_modus(X_train, y_train)

# --- Train Random Forest Fallback ---
if modus_results:
    st.subheader("Modus API Results")
    st.json(modus_results)
    classification_result = modus_results.get("classification_report", "No report available")
else:
    st.warning("Falling back to sklearn Random Forest...")
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    classification_result = classification_report(y_test, y_pred)

# Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)
data_ml['Cluster'] = clusters

# Merge with integrated data
data_with_clusters = pd.merge(
    integrated_data, data_ml[['symbol', 'Cluster', 'total_mutations']], on='symbol', how='left'
)
data_with_clusters['total_mutations'].fillna(0, inplace=True)

# --- Neo4j Integration ---
class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def create_schema_and_insert_data(self, data):
        with self.driver.session() as session:
            session.execute_write(self._create_schema_and_insert_data, data)
    
    @staticmethod
    def _create_schema_and_insert_data(tx, data):
        for _, row in data.iterrows():
            related_genes = data[data['Cluster'] == row['Cluster']]['symbol'].tolist()
            related_genes.remove(row['symbol'])
            relationship_strength = row.get('total_mutations', 0)
            query = """
            MERGE (gene:Gene {gene_id: $gene_id})
            ON CREATE SET
                gene.symbol = $symbol,
                gene.cluster = $cluster
            FOREACH (related_symbol IN $related_genes |
                MERGE (related:Gene {symbol: related_symbol})
                MERGE (gene)-[:RELATED_TO {weight: $relationship_strength}]->(related)
            )
            """
            tx.run(query, {
                "gene_id": row['gene_id'],
                "symbol": row['symbol'],
                "cluster": row['Cluster'],
                "relationship_strength": relationship_strength,
                "related_genes": related_genes
            })

# Insert into Neo4j
neo4j_handler = Neo4jHandler(neo4j_uri, neo4j_user, neo4j_password)
neo4j_handler.create_schema_and_insert_data(data_with_clusters)
neo4j_handler.close()

def fetch_graph_data_from_neo4j(cluster=None, gene=None):
    query = """
    MATCH (gene:Gene)-[r:RELATED_TO]->(related:Gene)
    {filter_clause}
    RETURN gene.symbol AS source, related.symbol AS target, r.weight AS weight
    """
    filter_clause = ""
    params = {}

    if cluster is not None:
        filter_clause = "WHERE gene.cluster = $cluster"
        params = {"cluster": cluster}
    elif gene is not None:
        filter_clause = "WHERE gene.symbol = $gene"
        params = {"gene": gene}

    query = query.replace("{filter_clause}", filter_clause)
    graph_data = {"nodes": [], "edges": []}

    with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)).session() as session:
        result = session.run(query, params)
        for record in result:
            graph_data["nodes"].append({"id": record["source"], "label": record["source"]})
            graph_data["nodes"].append({"id": record["target"], "label": record["target"]})
            graph_data["edges"].append({"from": record["source"], "to": record["target"], "weight": record["weight"] or 1})

    return graph_data

# --- Visualization ---
def visualize_with_pyvis(cluster=None, gene=None):
    net = Network(height="750px", width="100%", notebook=False)
    query = """
    MATCH (gene:Gene)-[r:RELATED_TO]->(related:Gene)
    {filter_clause}
    RETURN gene.symbol AS source, related.symbol AS target, r.weight AS weight
    """
    filter_clause = ""
    params = {}

    if cluster is not None:
        filter_clause = "WHERE gene.cluster = $cluster"
        params = {"cluster": cluster}
    elif gene is not None:
        filter_clause = "WHERE gene.symbol = $gene"
        params = {"gene": gene}

    query = query.replace("{filter_clause}", filter_clause)
    node_colors = {}

    with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)).session() as session:
        result = session.run(query, params)
        for record in result:
            source = record["source"]
            target = record["target"]
            weight = record["weight"] or 1
            edge_thickness = max(1, weight / 10)

            if source not in node_colors:
                node_colors[source] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            if target not in node_colors:
                node_colors[target] = "#{:06x}".format(random.randint(0, 0xFFFFFF))

            net.add_node(source, label=source, color=node_colors[source])
            net.add_node(target, label=target, color=node_colors[target])
            net.add_edge(source, target, width=edge_thickness)

    net.write_html("graph.html")
    return "graph.html"

# Streamlit UI
st.title("Prostate Cancer Gene Analysis")
st.subheader("Insights from Modus API and Neo4j Visualization")
st.subheader("Clustered data and relationships between genes are visualized interactively")
st.subheader("Insights like relationships between genes in the same cluster can help researchers identify co-mutations or new targets")

st.subheader("Classification Report")
st.text(classification_result)

st.subheader("Clustered Data")
st.dataframe(data_ml)

st.header("Graph Visualization")
mode = st.radio("Visualization Mode", ["Cluster", "Gene"])
if mode == "Cluster":
    cluster_to_query = st.number_input("Cluster to visualize", min_value=0, max_value=data_ml['Cluster'].max(), value=0)
    if st.button("Visualize Cluster"):
        graph_file = visualize_with_pyvis(cluster=cluster_to_query)
        with open(graph_file, "r") as f:
            st.components.v1.html(f.read(), height=800)
elif mode == "Gene":
    gene_to_query = st.selectbox("Gene to visualize", data_with_clusters['symbol'].unique())
    if st.button("Visualize Gene"):
        graph_file = visualize_with_pyvis(gene=gene_to_query)
        with open(graph_file, "r") as f:
            st.components.v1.html(f.read(), height=800)

# Export options
st.header("Export Options")
st.subheader("Download and share for collaboration purposes")
if st.button("Export Clustered Data"):
    csv_data = data_ml.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Clustered Data CSV", data=csv_data, file_name="clustered_data.csv", mime="text/csv")

if st.button("Export Neo4j Graph as JSON"):
    graph_data = fetch_graph_data_from_neo4j()  # Adjust filters if needed
    json_data = json.dumps(graph_data, indent=4).encode('utf-8')
    st.download_button(label="Download Graph JSON", data=json_data, file_name="graph.json", mime="application/json")
