# topic_modeling_by_year.py

import os
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# === Load CSV and group by year ===
df = pd.read_csv("./esg_corpus_by_year.csv")
grouped = df.groupby("year")

# === Create output directory ===
output_dir = "./topic_models"
os.makedirs(output_dir, exist_ok=True)

# === BERTopic per year ===
def run_bertopic_by_year():
    for year, group in grouped:
        docs = group["content_raw"].dropna().tolist()
        if len(docs) < 15:  # Further increased minimum docs requirement
            print(f"[Skip] Not enough docs for BERTopic in year {year} (need at least 15, got {len(docs)})")
            continue
        print(f"[BERTopic] Running for year {year} with {len(docs)} documents")
        
        try:
            # More conservative BERTopic parameters for small datasets
            topic_model = BERTopic(
                language="english", 
                calculate_probabilities=False,  # Disable probability calculation to avoid issues
                min_topic_size=3,  # Increase minimum topic size
                nr_topics=5,       # Set fixed number of topics
                verbose=True       # Enable verbose output for debugging
            )
            topics, probs = topic_model.fit_transform(docs)
            topic_model.save(f"{output_dir}/bertopic_model_{year}")
            
            # Get the actual number of topics found
            n_topics = len(topic_model.get_topics())
            print(f"[BERTopic] Found {n_topics} topics for year {year}")
            
            # Only create visualization if we have multiple topics
            if n_topics > 1:
                # Use min of actual topics or 10
                top_n = min(n_topics, 10)
                try:
                    topic_model.visualize_barchart(top_n_topics=top_n).write_html(f"{output_dir}/bertopic_{year}_barchart.html")
                    print(f"[BERTopic] Created barchart with {top_n} topics for year {year}")
                except Exception as e:
                    print(f"[BERTopic] Error creating barchart for year {year}: {e}")
            else:
                print(f"[BERTopic] Only {n_topics} topic(s) found for year {year}, skipping barchart visualization")
                
        except Exception as e:
            print(f"[BERTopic] Error processing year {year}: {e}")
            print(f"[BERTopic] Error details: {type(e).__name__}")
            continue

# === LDA per year ===
def run_lda_by_year(n_topics=5):
    for year, group in grouped:
        docs = group["content_clean"].dropna().tolist()
        if len(docs) < 5:
            print(f"[Skip] Not enough docs for LDA in year {year}")
            continue
        print(f"[LDA] Running for year {year}")
        vectorizer = CountVectorizer(stop_words="english", max_df=0.95, min_df=2)
        dtm = vectorizer.fit_transform(docs)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)
        terms = vectorizer.get_feature_names_out()
        with open(f"{output_dir}/lda_{year}_topics.txt", "w") as f:
            for idx, topic in enumerate(lda.components_):
                top_terms = [terms[i] for i in topic.argsort()[:-11:-1]]
                f.write(f"Topic {idx}: {' | '.join(top_terms)}\n")

# === Run both ===
if __name__ == "__main__":
    run_bertopic_by_year()
    run_lda_by_year(n_topics=5)
