import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from hybrid_esg_retriever_claude_final import hybrid_esg_answer

# === Load data ===
df_year = pd.read_csv("actual_keyword_trends_ngram.csv", index_col=0)
df_company = pd.read_csv("keyword_trends_by_company.csv", index_col=0)
df_industry = pd.read_csv("keyword_trends_by_industry.csv", index_col=0)

# === Page Config ===
st.set_page_config(page_title="ESG Insight Engine", layout="wide")
st.title("🌱 ESG Trend Insight Engine")
st.caption("Powered by GenAI + Knowledge Graphs | Developed by [Wan-Ning (Winnie) Chen]")

# === Tabbed UI ===
tab1, tab2 = st.tabs(["📊 Keyword Trend Explorer", "🧠 ESG QA + Explainability"])

# === TAB 1: Keyword Explorer ===
with tab1:
    st.header("📊 Keyword Frequency Trends")

    group_type = st.selectbox("Group by", ["Year", "Company", "Industry"])
    if group_type == "Year":
        df = df_year
    elif group_type == "Company":
        df = df_company
    else:
        df = df_industry

    keywords = df.index.tolist()
    selected_keywords = st.multiselect("Select ESG keywords", keywords, default=keywords[:5])

    if selected_keywords:
        fig, ax = plt.subplots(figsize=(10, 6))
        for kw in selected_keywords:
            if kw in df.index:
                ax.plot(df.columns, df.loc[kw], label=kw, marker="o")
        ax.set_title(f"Keyword Trends by {group_type}", fontsize=14)
        ax.set_xlabel(group_type)
        ax.set_ylabel("Mentions")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Please select at least one keyword.")

# === TAB 2: ESG QA + Summarization ===
with tab2:
    st.header("🧠 ESG Question Answering (Hybrid Retrieval + Claude)")

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_company = st.selectbox("Select Company", df_company.columns)
    with col2:
        selected_year = st.selectbox("Select Year", df_year.columns)
    with col3:
        selected_topic = st.selectbox("Select Topic", df_year.index.tolist())

    explain_toggle = st.checkbox("🔍 Show KG + Vector Context", value=True)

    if st.button("Generate ESG Insight"):
        with st.spinner("Generating ESG summary with hybrid engine..."):
            result = hybrid_esg_answer(selected_company, selected_topic, selected_year)

            st.subheader("📌 User Question")
            st.code(result["question"])

            if explain_toggle:
                st.subheader("📘 Knowledge Graph Facts")
                st.text_area("KG Facts", result["kg_context"], height=150)

                st.subheader("📄 Vector Search Context")
                st.text_area("Vector Context", result["vector_snippet"], height=150)

            st.subheader("🧠 Claude ESG Summary")
            st.success(result["summary"])
