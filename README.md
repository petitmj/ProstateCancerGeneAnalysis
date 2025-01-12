# Prostate Cancer Gene Analysis

This project provides an interactive web-based platform for analyzing prostate cancer gene data using Neo4j for knowledge graph representation, PyVis for visualization, and Modus API for machine learning analysis (with fallback to `sklearn` Random Forest).

## Features

1. **Machine Learning Analysis**
   - Integrates **Modus API** for feature analysis and prediction.
   - Classification of genes based on mutation frequency.

2. **Clustering**
   - Uses KMeans clustering to group genes based on mutation frequency.

3. **Knowledge Graph with Neo4j**
   - Visualizes relationships between genes using Neo4j.
   - Dynamically filters visualizations by gene or cluster.

4. **Graph Visualization**
   - Interactive PyVis graph visualization with:
     - **Unique colors** for each node.
     - **Edge thickness** proportional to relationship strength.

5. **Export Options**
   - Download clustered data as a CSV file.
   - Export graph relationships as a JSON file.

---

## Setup Instructions

### Prerequisites

- **Python 3.8+**
- Neo4j Aura (or a local Neo4j database)
- [Streamlit](https://streamlit.io)
- Modus API

### Clone the Repository
```bash
git clone https://github.com/<username>/prostate-cancer-gene-analysis.git
cd prostate-cancer-gene-analysis
```

### Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Neo4j Configuration

1. Set up a Neo4j Aura instance (or local Neo4j database).
2. Add your **Neo4j credentials** to the `secrets.toml` file:
   ```toml
   [secrets]
   NEO4J_URI = "neo4j+s://<your-neo4j-uri>"
   NEO4J_USER = "neo4j"
   NEO4J_PASSWORD = "<your-password>"
   ```

### Modus API Configuration (Optional)

1. Obtain your **Modus API URL** and **API Key**.
2. Add the details to `secrets.toml`:
   ```toml
   MODUS_API_URL = "https://api.modusplatform.com/analyze"
   MODUS_API_KEY = "<your-api-key>"
   ```

---

## Run the Application

To start the Streamlit app:
```bash
streamlit run app.py
```

---

## How to Use

### 1. Load and Preprocess Data
- Automatically filters prostate cancer-associated genes from:
  - `cancer_predisposition_variants.csv`
  - `gene_list.csv`

### 2. Perform Machine Learning
- If Modus API is available:
  - Predicts using Modus API.
- If unavailable:
  - Falls back to `sklearn`'s Random Forest Classifier.

### 3. View Results
- View the **classification report** and **clustered data** interactively.
- Identify co-mutated genes or targets through clusters.

### 4. Visualize Knowledge Graph
- **Filter by Cluster**: Visualize all relationships within a specific cluster.
- **Filter by Gene**: Visualize relationships for a specific gene.

### 5. Export Data
- Download clustered data as a CSV file.
- Export graph relationships as a JSON file.

---

## Key Files

| File                                    | Description                                        |
|-----------------------------------------|----------------------------------------------------|
| `app.py`                                | Main Streamlit application.                        |
| `requirements.txt`                      | Python dependencies.                               |
| `.streamlit/secrets.toml`               | Configuration file for secrets (e.g., Neo4j and Modus API credentials). |
| `data/cancer_predisposition_variants.csv` | Sample data for cancer predisposition variants.    |
| `data/gene_list.csv`                    | Sample gene list data.                             |

---

## Technical Details

### Machine Learning
- Uses Modus API.
- Automatically handles missing data for Neo4j insertion.

### Knowledge Graph
- Built using Neo4j.
- `RELATED_TO` relationships include a **weight** property proportional to mutation frequency.

### Visualization
- PyVis-based interactive graphs:
  - **Node colors** are randomly assigned.
  - **Edge thickness** varies based on mutation frequency.

---

## Sample Outputs

### 1. Clustered Data
| Symbol   | Cluster | Total Mutations |
|----------|---------|-----------------|
| BRCA1    | 0       | 12              |
| BRCA2    | 1       | 8               |

### 2. Graph Visualization
Interactive PyVis graph showing nodes (genes) and edges (relationships):
- **Thick edges** indicate strong relationships.

---

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

---

## Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.
`

