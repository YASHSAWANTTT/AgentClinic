# rag.py
from typing import List, Dict, Any
import csv, math

def load_guideline_snippets(path="data/guideline_snippets.csv") -> List[Dict[str, Any]]:
    docs = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # expect columns: source_id, section, text
            docs.append({"source_id": row["source_id"], "section": row.get("section",""), "text": row["text"]})
    return docs

class TinyBM25:
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        self.avg_len = max(1, sum(len(d["text"].split()) for d in docs) / max(1,len(docs)))
        self.df = {}
        for d in docs:
            for t in set(d["text"].lower().split()):
                self.df[t] = self.df.get(t, 0) + 1

    def _score(self, query: str, doc_text: str, k1=1.5, b=0.75) -> float:
        q_terms = query.lower().split()
        d_terms = doc_text.lower().split()
        dlen = max(1,len(d_terms))
        score = 0.0
        for t in q_terms:
            if t not in self.df: 
                continue
            df_t = self.df[t]
            idf = math.log(1 + (len(self.docs) - df_t + 0.5) / (df_t + 0.5))
            tf = d_terms.count(t)
            denom = tf + k1 * (1 - b + b * dlen / self.avg_len)
            score += idf * (tf * (k1 + 1)) / (denom + 1e-9)
        return score

    def topk(self, query: str, k=3):
        scored = [(self._score(query, d["text"]), d) for d in self.docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:k] if s > 0]

class GuidelineRAG:
    def __init__(self, path="data/guideline_snippets.csv"):
        self.docs = load_guideline_snippets(path)
        self.bm25 = TinyBM25(self.docs)

    def retrieve(self, query: str, k: int = 1):
        hits = self.bm25.topk(query, k=k)
        return [{"source_id": h["source_id"], "quote": h["text"][:280]} for h in hits]
