#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_trulens_with_chroma.py
- Evaluate retrieval quality directly against your persisted Chroma index.
- Uses the SAME embedding model for query as index time via query_embeddings.
- Outputs per-question metrics CSV + summary.
"""

import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Embeddings for local metrics AND for Chroma query_embeddings
from sentence_transformers import SentenceTransformer

# Chroma (same collection as your indexing script)
import chromadb

# ---------- Config: must match index script ----------
PERSIST_DIR = "./chroma_esg_index"
COLLECTION = "esg-cleaned-docs"
EMB_MODEL = "all-MiniLM-L6-v2"

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs("outputs", exist_ok=True)


@dataclass
class EvalItem:
    qid: str
    category: str
    question: str
    answer: str
    gold_doc_ids: List[str]


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str


@dataclass
class EvalResult:
    qid: str
    category: str
    question: str
    gold_doc_ids: str
    retrieved_ids: str
    hit_at_k: int
    recall_at_k: float
    answer_sim: float
    ctx_relevance_avg: float
    groundedness: float
    latency_ms: int


def load_eval_items(csv_path: str) -> List[EvalItem]:
    df = pd.read_csv(csv_path)
    items: List[EvalItem] = []
    for _, r in df.iterrows():
        g = []
        if "gold_doc_ids" in df.columns and isinstance(r.get("gold_doc_ids", ""), str):
            g = [s.strip() for s in str(r["gold_doc_ids"]).split(";") if s.strip()]
        items.append(EvalItem(
            qid=str(r.get("id", "")),
            category=str(r.get("category", "")),
            question=str(r.get("question", "")),
            answer=str(r.get("answer", "")),
            gold_doc_ids=g
        ))
    return items


def build_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION)
    return collection

def build_chroma_id_map(collection):
    id_map = {}
    custom_ids = set()
    data = collection.get(include=["metadatas"], limit=50000)
    ids = data.get("ids", [])
    metas = data.get("metadatas", [])
    for i, cid in enumerate(ids):
        md = metas[i] or {}
        # Prefer your custom key
        custom = (md.get("custom_doc_id") or
                  md.get("doc_id") or
                  md.get("document_id") or
                  md.get("ref_doc_id") or
                  md.get("source") or
                  md.get("file_path") or "")
        if custom:
            custom = str(custom)
            id_map[cid] = custom
            custom_ids.add(custom)
        else:
            id_map[cid] = cid
    return id_map, custom_ids

# def search_chroma(collection, query: str, top_k: int, encoder: SentenceTransformer) -> List[RetrievedDoc]:
#     # Use the SAME encoder as index-time, and pass query_embeddings to Chroma
#     q_emb = encoder.encode([query], convert_to_numpy=True)[0].tolist()
#     res = collection.query(
#         query_embeddings=[q_emb],
#         n_results=top_k,
#         include=["metadatas", "documents"]
#     )
#     docs: List[RetrievedDoc] = []
#     if res and res.get("documents"):
#         for i, doc_text in enumerate(res["documents"][0]):
#             md = res["metadatas"][0][i] if res.get("metadatas") else {}
#             doc_id = md.get("doc_id") or md.get("id") or f"doc-{i}"
#             docs.append(RetrievedDoc(doc_id=doc_id, text=doc_text))
#     return docs
# def search_chroma(collection, query: str, top_k: int, encoder, id_map):
#     q_emb = encoder.encode([query], convert_to_numpy=True)[0].tolist()
#     res = collection.query(query_embeddings=[q_emb],
#                            n_results=top_k,
#                            include=["metadatas","documents","ids"])
#     docs = []
#     if res and res.get("documents"):
#         for i, doc_text in enumerate(res["documents"][0]):
#             raw_id = res["ids"][0][i]
#             mapped = id_map.get(raw_id, raw_id)  # now maps to your custom_doc_id
#             docs.append(RetrievedDoc(doc_id=mapped, text=doc_text))
#     return docs
# correct version
# def search_chroma(collection, query: str, top_k: int, encoder, id_map):
#     q_emb = encoder.encode([query], convert_to_numpy=True)[0].tolist()
#     res = collection.query(
#         query_embeddings=[q_emb],
#         n_results=top_k,
#         include=["metadatas","documents"]
#     )
#     docs = []
#     if res and res.get("documents"):
#         for i, doc_text in enumerate(res["documents"][0]):
#             raw_id = res["ids"][0][i]
#             mapped = id_map.get(raw_id, raw_id)  # translate to your custom_doc_id if known
#             docs.append(RetrievedDoc(doc_id=mapped, text=doc_text))
#     return docs
# correct version end
def search_chroma(collection, query: str, top_k: int, encoder, id_map):
    q_emb = encoder.encode([query], convert_to_numpy=True)[0].tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["metadatas","documents"]
    )
    docs = []
    if res and res.get("documents"):
        for i, doc_text in enumerate(res["documents"][0]):
            raw_id = res["ids"][0][i]
            md = res["metadatas"][0][i] or {}
            # Prefer explicit custom_doc_id if you still want to keep it
            mapped = md.get("custom_doc_id") or id_map.get(raw_id, raw_id)
            docs.append(RetrievedDoc(doc_id=mapped, text=doc_text))
            # Attach the metadata so we can normalize (simple way: stash on object)
            docs[-1].metadata = md  # add this attribute dynamically
    return docs



def hit_at_k_fn(retrieved_ids: List[str], gold_ids: List[str]) -> int:
    return int(len(set(retrieved_ids).intersection(set(gold_ids))) > 0)


def recall_at_k_fn(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    return float(len(set(retrieved_ids).intersection(set(gold_ids))) / len(gold_ids)) if gold_ids else np.nan


def cosine_sim_txt(a: str, b: str, encoder: SentenceTransformer) -> float:
    vecs = encoder.encode([a, b], convert_to_numpy=True)
    return float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])


def avg_ctx_rel(question: str, contexts: List[RetrievedDoc], encoder: SentenceTransformer) -> float:
    if not contexts:
        return np.nan
    qv = encoder.encode([question], convert_to_numpy=True)
    cv = encoder.encode([c.text[:512] for c in contexts], convert_to_numpy=True)
    sims = cosine_similarity(qv, cv)[0]
    return float(np.mean(sims))


def groundedness(answer: str, contexts: List[RetrievedDoc], encoder: SentenceTransformer) -> float:
    if not contexts or not answer:
        return np.nan
    a = encoder.encode([answer], convert_to_numpy=True)
    c = encoder.encode([c.text[:512] for c in contexts], convert_to_numpy=True)
    sims = cosine_similarity(a, c)[0]
    return float(np.max(sims))


def run_eval(eval_csv: str, top_k: int, model_name: str, debug: int):
    items = load_eval_items(eval_csv)
    collection = build_collection()
    # checking start
    # print("[DEBUG] PERSIST_DIR:", os.path.abspath(PERSIST_DIR))
    # client = chromadb.PersistentClient(path=PERSIST_DIR)
    # print("[DEBUG] collections:", [c.name for c in client.list_collections()])
    # print("[DEBUG] using collection:", COLLECTION)

    # # Count how many records (nodes/chunks) are in the collection
    # try:
    #     print("[DEBUG] collection.count():", collection.count())
    # except Exception as e:
    #     print("[DEBUG] count() error:", e)

    # # Peek a few metadatas (do NOT request 'ids' in include on some versions)
    # sample = collection.get(include=["metadatas","documents"], limit=5)
    # print("[DEBUG] sample metadatas:", sample.get("metadatas", []))
    # print("[DEBUG] num documents returned:", len(sample.get("documents", [[]])[0]) if sample.get("documents") else 0)
    # checking end
    id_map, custom_ids = build_chroma_id_map(collection)  # ✅ build once
    encoder = SentenceTransformer(model_name)

    results: List[EvalResult] = []

    for idx, it in enumerate(tqdm(items, desc="Evaluating (Chroma embeddings)")):
        t0 = time.time()
        # original version
        # ctxs = search_chroma(collection, it.question, top_k, encoder, id_map)  # ✅ pass id_map
        # retrieved_ids = [c.doc_id for c in ctxs]

        # # Metrics
        # h = hit_at_k_fn(retrieved_ids, it.gold_doc_ids) if it.gold_doc_ids else 0
        # r = recall_at_k_fn(retrieved_ids, it.gold_doc_ids) if it.gold_doc_ids else np.nan
        # orginal version end
        
        ctxs = search_chroma(collection, it.question, top_k, encoder, id_map)
        retrieved_ids = [c.doc_id for c in ctxs]

        # --- Normalized-by-family Hit/Recall (company_year) ---
        gold_families = gold_to_family_keys(it.gold_doc_ids)

        ret_families = set()
        for c in ctxs:
            md = getattr(c, "metadata", {}) or {}
            fam = meta_to_family_key(md)              # prefers metadata {company, year}
            if not fam:                               # fallback: infer from retrieved custom_doc_id
                parsed = gold_to_family_keys([c.doc_id])
                fam = next(iter(parsed), "")
            if fam:
                ret_families.add(fam)

        h = 1 if gold_families and (gold_families & ret_families) else 0
        r = (len(gold_families & ret_families) / len(gold_families)) if gold_families else float("nan")
        # --- end normalized Hit/Recall ---


        ans_sim = cosine_sim_txt(it.answer, " ".join([c.text[:300] for c in ctxs]), encoder) if it.answer else np.nan
        ctx_rel = avg_ctx_rel(it.question, ctxs, encoder)
        grd = groundedness(it.answer, ctxs, encoder)

        latency_ms = int((time.time() - t0) * 1000)

        results.append(EvalResult(
            qid=it.qid,
            category=it.category,
            question=it.question,
            gold_doc_ids=";".join(it.gold_doc_ids),
            retrieved_ids=";".join(retrieved_ids),
            hit_at_k=h,
            recall_at_k=r,
            answer_sim=ans_sim,
            ctx_relevance_avg=ctx_rel,
            groundedness=grd,
            latency_ms=latency_ms
        ))

        # Debug print first N
        if debug > 0 and idx < debug:
            print(f"\n[DEBUG] QID={it.qid}")
            print("  Question:", it.question)
            print("  GOLD:", it.gold_doc_ids)
            print("  RET :", retrieved_ids[:top_k])

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"outputs/eval_run_chroma_{ts}.csv"
    pd.DataFrame([r.__dict__ for r in results]).to_csv(out_path, index=False)
    print(f"\n[OK] Saved: {out_path}")

    # Summary
    hit_rate = float(np.mean([r.hit_at_k for r in results])) if results else 0.0
    recall_vals = [r.recall_at_k for r in results if isinstance(r.recall_at_k, float) and not np.isnan(r.recall_at_k)]
    avg_recall = float(np.mean(recall_vals)) if recall_vals else float("nan")
    avg_ans_sim = float(np.nanmean([r.answer_sim for r in results])) if results else float("nan")
    avg_ctx_rel_summary = float(np.nanmean([r.ctx_relevance_avg for r in results])) if results else float("nan")
    avg_grounded = float(np.nanmean([r.groundedness for r in results])) if results else float("nan")

    print("\n=== Baseline (Chroma) Summary ===")
    print(f"Questions:     {len(results)}")
    print(f"Hit@{top_k}:     {hit_rate:.3f}")
    print(f"Recall@{top_k}:  {avg_recall:.3f}" if not np.isnan(avg_recall) else "Recall@K:  N/A")
    print(f"AnsSimilarity: {avg_ans_sim:.3f}")
    print(f"CtxRelevance:  {avg_ctx_rel_summary:.3f}")
    print(f"Groundedness:  {avg_grounded:.3f}")
    print("=================================\n")

# add normalize function
import re

def gold_to_family_keys(gold_ids: list[str]) -> set[str]:
    """
    Convert gold custom_doc_id strings like 'Apple_2023-631795220' or 'Apple-367185597'
    into normalized family keys 'Apple_2023' (or 'Apple' if year missing).
    """
    keys = set()
    for gid in gold_ids or []:
        m = re.match(r"^([A-Za-z]+)[\-_]?(20\d{2})?", str(gid))
        if m:
            company = m.group(1)
            year = m.group(2) or ""  # allow missing year
            key = f"{company}_{year}" if year else company
            keys.add(key)
    return keys

def meta_to_family_key(md: dict) -> str:
    """
    Build family key for a retrieved hit using metadata (prefer precise fields).
    """
    company = (md.get("company") or md.get("org") or "").strip()
    year = str(md.get("year") or "").strip()
    if year and re.match(r"^20\d{2}$", year):
        return f"{company}_{year}" if company else year
    return company or ""  # fallback
# normalize function end


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", type=str, default="eval_set.csv")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--model", type=str, default=EMB_MODEL)
    ap.add_argument("--debug", type=int, default=3, help="print GOLD vs RET for first N questions (0 to disable)")
    args = ap.parse_args()
    run_eval(eval_csv=args.eval_csv, top_k=args.top_k, model_name=args.model, debug=args.debug)
