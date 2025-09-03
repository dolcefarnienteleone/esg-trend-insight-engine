import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# === Load precomputed keyword trends ===
df_year = pd.read_csv("actual_keyword_trends_ngram.csv", index_col=0)
df_company = pd.read_csv("keyword_trends_by_company.csv", index_col=0)
df_industry = pd.read_csv("keyword_trends_by_industry.csv", index_col=0)

# === App title ===
st.set_page_config(page_title="ESG Explorer", layout="wide")
st.title("🌿 ESG Keyword Explorer")

# === Sidebar controls ===
group_type = st.sidebar.selectbox("Group by", ["Year", "Company", "Industry"])
if group_type == "Year":
    df = df_year
elif group_type == "Company":
    df = df_company
else:
    df = df_industry

available_keywords = df.index.tolist()
selected_keywords = st.sidebar.multiselect("Select keywords to plot", available_keywords, default=available_keywords[:5])

# === Plot section ===
if selected_keywords:
    st.subheader(f"📈 Keyword Frequency by {group_type}")
    fig, ax = plt.subplots(figsize=(10, 6))
    for kw in selected_keywords:
        if kw in df.index:
            ax.plot(df.columns, df.loc[kw], label=kw, marker="o")
    ax.set_title(f"Keyword Trends by {group_type}")
    ax.set_xlabel(group_type)
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("Please select at least one keyword.")

# === Footer ===
st.caption("Powered by LDA topic modeling and document frequency analysis (NLP pipeline)")

# # === RAG + Claude Insight Generator ===
# from esg_retriever_summarizer import retrieve_and_summarize

# st.markdown("---")
# st.header("🧠 ESG RAG + Claude Insight Generator")

# col1, col2, col3 = st.columns(3)
# with col1:
#     selected_company = st.selectbox("Select Company", df_company.columns)
# with col2:
#     selected_year = st.selectbox("Select Year", df_year.columns)
# with col3:
#     selected_topic = st.selectbox("Select Topic", df_year.index.tolist())

# if st.button("Generate ESG Summary with Claude"):
#     with st.spinner("Retrieving and summarizing..."):
#         result = retrieve_and_summarize(selected_company, selected_topic, selected_year)
#         st.subheader("🔍 Query")
#         st.code(result["query"])
#         st.subheader("📄 Retrieved Text")
#         st.text_area("Retrieved Excerpt", result["raw_text"], height=250)
#         st.subheader("🧠 Claude Summary")
#         st.success(result["summary"])

# === RAG + Claude Insight Generator ===
from hybrid_esg_retriever_claude_final import hybrid_esg_answer

st.markdown("---")
st.header("🧠 ESG Summary + Explainability")

col1, col2, col3 = st.columns(3)
with col1:
    selected_company = st.selectbox("Select Company", df_company.columns)
with col2:
    selected_year = st.selectbox("Select Year", df_year.columns)
with col3:
    selected_topic = st.selectbox("Select Topic", df_year.index.tolist())

# Optional: Explanation toggle
explain_toggle = st.checkbox("🔍 Show Explainability (KG + Vector Context)", value=True)

if st.button("Generate ESG Insight"):
    with st.spinner("Generating summary with hybrid context..."):
        result = hybrid_esg_answer(selected_company, selected_topic, selected_year)

        st.subheader("📌 User Question")
        st.code(result["question"])

        if explain_toggle:
            st.subheader("📘 Knowledge Graph Facts")
            st.text_area("KG Facts", result["kg_context"], height=150)

            st.subheader("📄 Vector Search Context")
            st.text_area("Vector Result Snippet", result["vector_snippet"], height=150)

        st.subheader("🧠 Claude ESG Summary")
        st.success(result["summary"])

