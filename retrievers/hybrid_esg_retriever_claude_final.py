import os
from py2neo import Graph
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from anthropic import Anthropic
from huggingface_hub import InferenceClient
import chromadb
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Neo4j setup
graph = Graph("bolt://localhost:7689", auth=("neo4j", os.getenv("neo4j_pw")))

# ✅ Claude setup
claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ✅ LlamaIndex setup (vector retrieval)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_esg_index")
collection = chroma_client.get_collection("esg-cleaned-docs")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
vector_query_engine = vector_index.as_query_engine()

# === Helper: KG Cypher lookup ===
def get_kg_facts(company, keyword, year):
    query = f"""
    MATCH (c:Company {{name: "{company}"}})-[r:MENTIONS]->(t:Topic)-[:IN_YEAR]->(y:Year {{value: {year}}})
    WHERE toLower(t.name) CONTAINS toLower("{keyword}")
    RETURN DISTINCT c.name AS company, t.name AS topic, y.value AS year, r.label AS label
    LIMIT 50
    """
    results = graph.run(query).data()
    return results

# === Combine KG + Vector ===
def hybrid_esg_answer(company, keyword, year):
    user_question = f"What did {company} say about {keyword} in {year}?"

    # 🔎 Vector search
    vector_result = vector_query_engine.query(user_question)

    # 🧠 KG search
    kg_results = get_kg_facts(company, keyword, year)
    kg_context = "\n".join([
        f"- {r['company']} mentioned '{r['topic']}' via '{r['label']}' in {r['year']}"
        for r in kg_results
    ]) or "(No structured facts found from KG)"

    # 🧩 Combine into Claude prompt
    prompt = f"""
You are an ESG assistant. Use the following retrieved evidence to answer the user’s question.

📘 Knowledge Graph Facts:
{kg_context}

📄 Vector Search Insight:
{str(vector_result)}

User Question: {user_question}

Please provide a detailed, fact-grounded ESG insight. If unsure, state that.
"""

    response = claude.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=500,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "question": user_question,
        "summary": response.content[0].text.strip(),
        "kg_context": kg_context,
        "vector_snippet": str(vector_result)
    }
