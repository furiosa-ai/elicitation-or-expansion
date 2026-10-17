"""BM25 retrieval service over the HotpotQA wiki17 abstracts (5.23M passages).

API-compatible with the DSPy ColBERTv2 server that GEPA/LangProBe used, so `dspy.ColBERTv2(url=...)`
clients also work:
    GET  /wiki17_abstracts?query=<q>&k=7   -> {"topk": [{"text": "Title | abstract", "pid": int, "score": float, "title": str}, ...]}
    POST /search  {"queries": [...], "k": 7} -> {"results": [[...], ...]}
    GET  /health

Run:  python -m domains.hotpotqa.retriever_server --index data/index/wiki17_bm25 --collection data/raw/wiki17/wiki17_abstracts.tsv --port 8001
"""

from __future__ import annotations

import argparse
import threading
import time

import bm25s
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI(title="wiki17 BM25 retriever")
_STATE: dict = {}
_LOCK = threading.Lock()


def load(index_dir: str, collection_tsv: str) -> None:
    t0 = time.time()
    retriever = bm25s.BM25.load(index_dir, mmap=False, load_corpus=False)
    retriever.backend = "numba"
    retriever.activate_numba_scorer()
    titles: list[str] = []
    texts: list[str] = []
    with open(collection_tsv, encoding="utf-8") as f:
        for line in f:
            pid, title, text = line.rstrip("\n").split("\t", 2)
            titles.append(title)
            texts.append(text)
    assert len(titles) == retriever.scores["num_docs"], f"collection/index size mismatch: {len(titles)} vs {retriever.scores['num_docs']}"
    _STATE.update(retriever=retriever, titles=titles, texts=texts)
    # warm numba
    retriever.retrieve(bm25s.tokenize(["warm up query"], stopwords="en", show_progress=False), k=3, show_progress=False, n_threads=1)
    print(f"loaded {len(titles)} passages in {time.time() - t0:.1f}s", flush=True)


def _tokenize(queries: list[str]) -> list[list[str]]:
    """Per-query token lists; queries that reduce to nothing (stopwords/digits/empty) yield []."""
    toks = bm25s.tokenize([q if q and q.strip() else "" for q in queries], stopwords="en", return_ids=False, show_progress=False)
    return [list(t) for t in toks]


def search(queries: list[str], k: int) -> list[list[dict]]:
    r = _STATE["retriever"]
    k = max(1, min(int(k), 100))
    toks = _tokenize(queries)
    nonempty = [i for i, t in enumerate(toks) if t]
    out: list[list[dict]] = [[] for _ in queries]
    if not nonempty:
        return out  # e.g. query "4" or "the": no scorable tokens -> deterministic empty result
    with _LOCK:  # numba scorer is not guaranteed re-entrant; queries are ~7ms so serialization is fine
        docs, scores = r.retrieve([toks[i] for i in nonempty], k=k, show_progress=False, n_threads=1)
    for j, i in enumerate(nonempty):
        hits = []
        for pid, sc in zip(docs[j], scores[j]):
            pid = int(pid)
            if sc <= 0:
                continue
            hits.append({"text": f"{_STATE['titles'][pid]} | {_STATE['texts'][pid]}", "pid": pid, "score": float(sc), "title": _STATE["titles"][pid]})
        out[i] = hits
    return out


@app.get("/health")
def health():
    return {"ok": "retriever" in _STATE, "num_docs": len(_STATE.get("titles", []))}


@app.get("/wiki17_abstracts")
def colbert_compat(query: str = Query(...), k: int = Query(7)):
    t0 = time.time()
    hits = search([query], k)[0]
    return {"topk": hits, "latency": round((time.time() - t0) * 1000, 1)}


class SearchRequest(BaseModel):
    queries: list[str]
    k: int = 7


@app.post("/search")
def batch_search(req: SearchRequest):
    return {"results": search(req.queries, req.k)}


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    load(a.index, a.collection)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", workers=1)


if __name__ == "__main__":
    main()
