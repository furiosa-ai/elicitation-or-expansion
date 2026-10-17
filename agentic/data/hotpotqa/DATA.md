# HotpotQA data

No question text, answers, passages or indexes are included. Rebuild them from the public sources; all commands
run from `agentic/`.

| File | Content |
|---|---|
| `split-tasks.csv` | Public IDs of the fixed splits in order: train 150, validation 300, selection 300, test 300 |
| `protocol.json` | Sampling recipe and split roles |

1. Export the Hugging Face `hotpot_qa` (`fullwiki`) train split in its original order:

   ```bash
   uv run --with datasets python -c "from datasets import load_dataset; \
   load_dataset('hotpot_qa', 'fullwiki', split='train').to_json('data/raw/hotpot_qa/fullwiki_train.jsonl')"
   ```

2. Build and verify the splits (`create_run.py` reads them from here):

   ```bash
   python -B domains/hotpotqa/data.py --raw data/raw/hotpot_qa/fullwiki_train.jsonl --out data/splits --test-out data/splits_test
   python -B data/hotpotqa/prepare_ids.py verify --tasks data/hotpotqa/split-tasks.csv \
     --train data/splits/hotpotqa/train.jsonl --val data/splits/hotpotqa/val.jsonl \
     --selection data/splits_test/hotpotqa/selection.jsonl --test data/splits_test/hotpotqa/test.jsonl
   ```

3. Build the retrieval corpus (`enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2` from
   https://nlp.stanford.edu/projects/hotpotqa/) and its BM25 index (k1 0.9, b 0.4), then serve it on an address
   the proposer container can reach:

   ```bash
   python -B data/hotpotqa/extract_collection.py \
     data/raw/wiki17/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 data/raw/wiki17/wiki17_abstracts.tsv
   uv run python -B data/hotpotqa/build_bm25.py data/raw/wiki17/wiki17_abstracts.tsv data/index/wiki17_bm25
   uv run python -m domains.hotpotqa.retriever_server --index data/index/wiki17_bm25 \
     --collection data/raw/wiki17/wiki17_abstracts.tsv --port 8001 --host 0.0.0.0
   ```
