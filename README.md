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

```mermaid

flowchart TD
    subgraph UI["Streamlit Frontend"]
        A[User interface to explore trends<br/>and ask questions]
    end

    subgraph Vector["LlamaIndex + ChromaDB"]
        B[Semantic vector search<br/>over ESG documents]
    end

    subgraph LLM["Claude API (Anthropic)"]
        C[LLM summarization<br/>of retrieved results]
    end

    subgraph KG["Neo4j Knowledge Graph"]
        D[Structured ESG triples<br/>(goals, topics, timelines)]
    end

    %% Pipeline flow
    A --> B
    A --> D
    B --> C
    D --> C

```

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

## 🎬 Demo Video

[![ESG Trend Engine Demo](https://github.com/dolcefarnienteleone/esg-trend-insight-engine/raw/main/media/esg_engine_thumbnail.jpg)](https://github.com/dolcefarnienteleone/esg-trend-insight-engine/raw/main/media/esg_trend_insight_engine.MP4)

---

## 📂 Folder Structure

```Kotlin
esg-trend-insight-engine/
├── data/
│ ├── esg_corpus_by_year.csv
│ ├── keyword_trends_by_*.csv
│ └── lda_*_topics.txt
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
```

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

## 🔐 Environment Variables

Create a `.env` file in the project root:

```ini
ANTHROPIC_API_KEY=your_claude_key
HUGGINGFACE_API_KEY=your_hf_key
neo4j_pw=your_neo4j_password
```

---

## 🚀 Deploy to Streamlit Cloud (Optional)

1. Push this repo to GitHub.
2. Go to **Streamlit Cloud** → **New app** → select your repo/branch.
3. Set **Main file** to `esg_explorer_app.py`.
4. Add Secrets (Environment variables):

- `ANTHROPIC_API_KEY`
- `HUGGINGFACE_API_KEY`
- `neo4j_pw`

⚠️ If you use Neo4j locally → switch to a hosted Neo4j instance or disable KG features for the cloud demo.

👉 **Live App**: https://your-app-name.streamlit.app

---

## 👤 Author

**Winnie Chen**
Data Scientist · NLP · ESG Analytics

🔗 [LinkedIn](https://www.linkedin.com/in/wanningchen)
🌐 [Portfolio](https://dolcefarnienteleone.github.io/#)

---

## 🧠 Inspiration

This project aims to make corporate ESG commitments more transparent, comparable, and explainable by combining:

- Topic modeling & keyword trend analysis

- Retrieval (LlamaIndex + ChromaDB)

- Reasoning & summarization (Claude)

- Structured facts (Neo4j Knowledge Graph)
