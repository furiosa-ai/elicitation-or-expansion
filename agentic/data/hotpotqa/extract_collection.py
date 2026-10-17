"""Extract HotpotQA wiki abstracts tarball into a flat TSV collection: pid \t title \t text.
Passage text mirrors the DSPy ColBERTv2 wiki17_abstracts server format: "Title | abstract".
"""
import bz2, json, sys, tarfile, time
src = sys.argv[1]; dst = sys.argv[2]
t0 = time.time(); n = 0; last_report = 0
with tarfile.open(src, "r:bz2") as tar, open(dst, "w", encoding="utf-8") as out:
    for m in tar:
        if not m.isfile() or not m.name.endswith(".bz2"):
            continue
        data = bz2.decompress(tar.extractfile(m).read())
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            title = d.get("title", "").replace("\t", " ").replace("\n", " ").strip()
            text = d.get("text", [])
            text = "".join(text) if isinstance(text, list) else str(text)
            text = text.replace("\t", " ").replace("\n", " ").strip()
            if not title or not text:
                continue
            out.write(f"{n}\t{title}\t{text}\n"); n += 1
        if n - last_report >= 250000:
            last_report = n
            print(f"progress {n} docs, {time.time()-t0:.0f}s, member={m.name}", flush=True)
print(f"DONE {n} docs in {time.time()-t0:.0f}s", flush=True)
open(dst + ".done", "w").write(str(n))
