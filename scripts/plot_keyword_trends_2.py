import pandas as pd
import matplotlib.pyplot as plt
from keyword_frequency_analyzer import compute_keyword_frequency

# === Load corpus ===
df = pd.read_csv("./esg_corpus_by_year.csv")

# === Define ESG-related keywords ===
keywords = [# 🔵 Environmental
    "carbon", "emissions", "climate", "energy", "renewable", "pollution",
    "co2", "greenhouse gas", "carbon footprint", "net zero", "recycling",

    # 🟠 Social
    "labor", "wages", "diversity", "equality", "inclusion", "workforce", "child labor",

    # 🟢 Governance
    "governance", "ethics", "transparency", "reporting", "board", "compliance",

    # 🔁 ESG Buzzwords
    "supply chain", "stakeholder", "sustainability", "esg", "circular economy"]

# === Frequency table by year ===
df_freq = compute_keyword_frequency(df, keywords, group_field="year")

# === Save frequency data ===
df_freq.to_csv("actual_keyword_trends_ngram.csv")

# === Plot ===
plt.figure(figsize=(12, 6))
for kw in df_freq.index:
    plt.plot(df_freq.columns, df_freq.loc[kw], label=kw, marker="o")

plt.title("Actual ESG Keyword Frequency by Year")
plt.xlabel("Year")
plt.ylabel("Frequency")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("actual_keyword_trend_plot.png")
plt.show()
