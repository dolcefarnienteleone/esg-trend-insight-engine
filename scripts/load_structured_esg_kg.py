import os
import json
import re
from py2neo import Graph, Node, Relationship
from dotenv import load_dotenv

load_dotenv()

# Neo4j connection
graph = Graph("bolt://localhost:7689", auth=("neo4j", os.getenv("neo4j_pw")))

# Folder containing extracted triple JSONs
TRIPLE_DIR = "./triples_output"

# Known company list for typed nodes
company_list = ["Apple", "Microsoft", "Google", "Samsung", "Sony",
                "FastRetailing", "Inditex", "Tesla", "JPMorgan", "BlackRock"]

def is_meaningful_topic(text):
    if not text:
        return False
    text = text.lower().strip()

    # Drop known generic or time-related phrases
    stoplist = {
        "annually", "monthly", "weekly", "report", "percent", "number",
        "yearly", "data", "amount", "figure", "increase", "decrease"
    }

    if text in stoplist:
        return False
    if text.isdigit():
        return False
    if len(text) < 4:
        return False
    if len(text.split()) < 2:
        return False  # prefer phrases like "carbon neutrality"
    return True


def extract_year_from_filename(filename):
    """Try to infer year from filename like 'Apple_ESG_Report_2021.txt'."""
    for token in filename.split("_"):
        if token.isdigit() and len(token) == 4:
            return int(token)
    # 如果找不到年份，返回預設值 2020
    return 2024

def clean_text(text):
    if not text:
        return "Unknown"
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:500]

def create_or_get_node(label, key, value):
    node = Node(label, **{key: value})
    graph.merge(node, label, key)
    return node

# === Main Loader ===
def load_typed_triples():
    total = 0
    for file in os.listdir(TRIPLE_DIR):
        if not file.endswith("_triples.json"):
            continue

        path = os.path.join(TRIPLE_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            triples = json.load(f)

        year = extract_year_from_filename(file)
        # 確保年份不為 None
        if year is None:
            year = 2020
        year_node = create_or_get_node("Year", "value", year)

        company_name = file.split("_")[0]
        if company_name not in company_list:
            continue
        company_node = create_or_get_node("Company", "name", company_name)

        for triple in triples:
            subj = clean_text(triple["subject"])
            pred = clean_text(triple["predicate"])
            obj = clean_text(triple["object"])

            # Skip if subject/object too short
            if len(subj) < 2 or len(obj) < 2:
                continue

            # topic_node = create_or_get_node("Topic", "name", obj)
            if not is_meaningful_topic(obj):
                continue  # Skip weak/irrelevant topics

            topic_node = create_or_get_node("Topic", "name", obj)

            # Create MENTIONS + IN_YEAR
            mentions_rel = Relationship(company_node, "MENTIONS", topic_node, label=pred)
            in_year_rel = Relationship(topic_node, "IN_YEAR", year_node)

            try:
                graph.merge(mentions_rel)
                graph.merge(in_year_rel)
                total += 1
            except Exception as e:
                print(f"Failed to add triple: {subj} - {pred} - {obj}: {e}")
                continue

    print(f"✅ Loaded {total} structured triples into Neo4j using typed schema")

if __name__ == "__main__":
    load_typed_triples()
