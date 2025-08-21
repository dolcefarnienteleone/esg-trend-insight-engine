# 🌱 ESG Trend Insight Engine

**End-to-end NLP pipeline to uncover ESG insights across companies and time.**
Combines GenAI (Claude), LLM RAG, Knowledge Graphs, and topic modeling for explainable ESG analysis.

---

## 🚀 Features

- 📊 **Keyword Trend Explorer**
  Visualize frequency of ESG themes (carbon, labor, governance, etc.) across time, companies, and industries.

- 🧠 **RAG-based ESG QA**
  Ask: _“What did Microsoft say about carbon in 2023?”_
  → Hybrid answers from both vector search and knowledge graph facts.

- 🔍 **Explainability Toggle**
  Toggle between:

  - Claude’s LLM summary
  - Neo4j-derived ESG facts
  - Retrieved document snippets

- 🧠 **Topic Modeling Layer**
  Discover top ESG themes per year using LDA and BERTopic.

---

## 🧱 Architecture

This project combines traditional NLP + modern retrieval + LLM summarization:

[Streamlit UI]
│
├──► LlamaIndex Vector Store (ChromaDB)
├──► Claude API (LLM summary)
└──► Neo4j Knowledge Graph (KG facts)

---

## 📊 Example Use Case

> **Q:** What were Microsoft’s ESG priorities in 2023?

**LLM Summary Output:**

- Carbon-free electricity investments
- Long-term carbon removal progress
- Focus on climate resilience

**KG Facts:**

- Microsoft AIMS_TO_ACHIEVE "carbon neutrality"
- Microsoft TARGET_YEAR "2030"

---

## 📂 Folder Structure

esg-trend-insight-engine/
├── data/
│ ├── esg*corpus_by_year.csv
│ ├── keyword_trends_by*.csv
│ └── lda\_.txt
├── scripts/
│ ├── triple_extractor.py
│ ├── load_structured_esg_kg.py
│ ├── topic_modeling_by_year.py
│ └── plot_keyword_trends.py
├── retrievers/
│ ├── hybrid_esg_retriever_claude.py
│ └── esg_kg_query_runner.py
├── esg_explorer_app.py
└── README.md

---

## ⚙️ Tech Stack

- **GenAI / RAG**: Claude 3 + LlamaIndex
- **Embeddings**: HuggingFace MiniLM
- **Vector DB**: ChromaDB
- **KG**: Neo4j (structured ESG facts)
- **Topic Modeling**: LDA, BERTopic
- **Frontend**: Streamlit

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/your-username/esg-trend-insight-engine.git
cd esg-trend-insight-engine
pip install -r requirements.txt
streamlit run esg_explorer_app.py
```
