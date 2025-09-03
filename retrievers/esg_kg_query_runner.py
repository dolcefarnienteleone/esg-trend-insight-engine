import os
from py2neo import Graph
from dotenv import load_dotenv

load_dotenv()
graph = Graph("bolt://localhost:7689", auth=("neo4j", os.getenv("neo4j_pw")))

# === Query 1: Companies that mentioned a topic in a specific year ===
def get_companies_by_topic_and_year(topic_keyword: str, year: int):
    query = f"""
    MATCH (c:Company)-[:MENTIONS]->(t:Topic)-[:IN_YEAR]->(y:Year {{value: {year}}})
    WHERE toLower(t.name) CONTAINS toLower("{topic_keyword}")
    RETURN DISTINCT c.name AS company, t.name AS topic
    ORDER BY company
    """
    return graph.run(query).to_data_frame()

# === Query 2: Topic evolution for one company ===
def get_company_topic_evolution(company_name: str):
    query = f"""
    MATCH (c:Company {{name: "{company_name}"}})-[:MENTIONS]->(t:Topic)-[:IN_YEAR]->(y:Year)
    RETURN y.value AS year, t.name AS topic
    ORDER BY year
    """
    return graph.run(query).to_data_frame()

# === Query 3: Top topics for all companies in a given year ===
def get_top_topics_by_year(year: int, top_n: int = 10):
    query = f"""
    MATCH (c:Company)-[:MENTIONS]->(t:Topic)-[:IN_YEAR]->(y:Year {{value: {year}}})
    RETURN t.name AS topic, COUNT(*) AS frequency
    ORDER BY frequency DESC
    LIMIT {top_n}
    """
    return graph.run(query).to_data_frame()

# === Query 4: Companies that started mentioning a topic after a year ===
def get_new_topic_mentions(topic_keyword: str, after_year: int):
    query = f"""
    MATCH (c:Company)-[:MENTIONS]->(t:Topic)-[:IN_YEAR]->(y:Year)
    WHERE toLower(t.name) CONTAINS toLower("{topic_keyword}") AND y.value > {after_year}
    RETURN DISTINCT c.name AS company, y.value AS year
    ORDER BY year
    """
    return graph.run(query).to_data_frame()

# === Query 5: Track mentions of a topic over time by company ===
def get_topic_mentions_by_company_and_year(topic_keyword: str):
    query = f"""
    MATCH (c:Company)-[:MENTIONS]->(t:Topic)-[:IN_YEAR]->(y:Year)
    WHERE toLower(t.name) CONTAINS toLower("{topic_keyword}")
    RETURN c.name AS company, y.value AS year, t.name AS topic
    ORDER BY company, year
    """
    return graph.run(query).to_data_frame()
