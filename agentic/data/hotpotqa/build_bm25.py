"""Build a bm25s index over the wiki17 abstracts collection and save it for mmap loading."""
import sys, time, resource, json
import bm25s
src, out = sys.argv[1], sys.argv[2]
t0 = time.time()
titles, docs = [], []
with open(src, encoding="utf-8") as f:
    for line in f:
        pid, title, text = line.rstrip("\n").split("\t", 2)
        titles.append(title); docs.append(f"{title} | {text}")
print(f"loaded {len(docs)} docs in {time.time()-t0:.0f}s", flush=True)
t0 = time.time()
tok = bm25s.tokenize(docs, stopwords="en", show_progress=False)
print(f"tokenized in {time.time()-t0:.0f}s", flush=True)
t0 = time.time()
ret = bm25s.BM25(k1=0.9, b=0.4)
ret.index(tok, show_progress=False)
print(f"indexed in {time.time()-t0:.0f}s", flush=True)
ret.save(out, corpus=None)
with open(f"{out}/titles.json", "w") as f:
    json.dump(titles, f)
print(f"saved to {out}; maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB", flush=True)
print("DONE", flush=True)
