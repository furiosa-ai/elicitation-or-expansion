# Elicitation or Expansion? Demystifying Gains from Agent Harness Optimization

<p align="center">
  Younghyun Kim<sup>1*</sup>, Wooseok Seo<sup>2*</sup>, Minseo Kim<sup>3</sup>, Seunghyuk Oh<sup>1</sup>, Minjae Lee<sup>1</sup>,<br>
  Donghoon Kim<sup>1</sup>, Hyung Il Koo<sup>1</sup>, Jaehyung Kim<sup>2&dagger;</sup>, Wonjun Kang<sup>1&dagger;</sup>
</p>
<p align="center">
  <sup>1</sup>FuriosaAI, <sup>2</sup>Yonsei University, <sup>3</sup>University of California, Berkeley<br>
  <sup>*</sup>Equal contribution, <sup>&dagger;</sup>Co-corresponding authors
</p>

<p align="center">
  <a href="https://furiosa-air.github.io/elicitation-or-expansion/"><img alt="Project" src="https://img.shields.io/static/v1?label=Project&message=Page&color=blue&logo=github-pages"></a>
  <!-- TODO: link the arXiv abstract and change the message to the arXiv ID once the preprint is out. -->
  <img alt="arXiv" src="https://img.shields.io/static/v1?label=arXiv&message=coming%20soon&color=lightgrey&logo=arxiv">
  <!-- TODO: link the Hugging Face dataset and change the message to "Hugging Face" once it is published. -->
  <img alt="Dataset" src="https://img.shields.io/static/v1?label=Dataset&message=coming%20soon&color=lightgrey&logo=huggingface">
</p>

This repository contains the code that runs the harness optimizations in the paper and the candidate harnesses
produced by the text-classification and HotpotQA runs. Generated AppWorld candidate snapshots are excluded because
their comments may contain protected benchmark material; the AppWorld optimization framework and public baseline
port are included. The rollouts are released separately as a Hugging Face dataset.

The proposer (Claude Opus 5 in Claude Code, effort max, auto-memory disabled) runs in an unprivileged Docker container
that sees only the run's checkout; the solver is any OpenAI-compatible endpoint (vLLM). Each run is 20 iterations with
two candidates per iteration. Included candidate harnesses are in `harnesses/<task>/<system>/`; they are not
inputs to a new run. The last column names the selected harness when its snapshot is included.
`harnesses/text_classification/baseline/fewshot_all_reroll.py` is the baseline harness of the text-classification
rollouts.

| Task | System | Configuration | Selected harness |
|---|---|---|---|
| Text classification | `id-20b-optimized` | `text_classification/config.yaml` (GPT-OSS-20B, medium effort) | `worked_case_memory.py` |
| HotpotQA | `8b-seed0` | `agentic/configs/hotpotqa-8b.yaml` | `linked_article_evidence.py` |
| HotpotQA | `8b-seed1` | `agentic/configs/hotpotqa-8b.yaml` with `--run-id hotpotqa-8b-seed1` | `graded_evidence_detail.py` |
| HotpotQA | `8b-seed2` | `agentic/configs/hotpotqa-8b.yaml` with `--run-id hotpotqa-8b-seed2` | `thinking_doubt_router.py` |
| HotpotQA | `8b-relaxed-seed0` | `agentic/configs/hotpotqa-8b-relaxed.yaml` | `shared_attribute_intersection.py` |
| HotpotQA | `8b-relaxed-seed1` | `agentic/configs/hotpotqa-8b-relaxed.yaml` with `--run-id hotpotqa-8b-relaxed-seed1` | `residual_query_probe.py` |
| HotpotQA | `8b-relaxed-seed2` | `agentic/configs/hotpotqa-8b-relaxed.yaml` with `--run-id hotpotqa-8b-relaxed-seed2` | `shared_property_intersection.py` |
| AppWorld | `120b-medium-seed0` | `agentic/configs/appworld-120b-medium.yaml` | Not included |
| AppWorld | `120b-medium-seed1` | `agentic/configs/appworld-120b-medium.yaml` with `--run-id appworld-120b-medium-seed1` | Not included |

`seedN` numbers independent optimization runs.

## Requirements

> [!NOTE]
> A run needs Docker (the proposer container), a Claude Code OAuth token for an account that can use Claude Opus 5,
> [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 for `text_classification/` and 3.12 for `agentic/`),
> and an OpenAI-compatible solver endpoint, for example vLLM on GPUs serving GPT-OSS-20B, Qwen3-8B or GPT-OSS-120B.
> HotpotQA also needs the public HotpotQA questions and Wikipedia abstracts and about 10 GB of memory to build the BM25
> index; AppWorld needs the AppWorld package and data, installed by `agentic/scripts/appworld_data_setup.sh`.
> No datasets are included.

## Layout

```
text_classification/      Meta-Harness text-classification example, run from a per-run copy
  meta_harness.py         optimization loop: propose, evaluate, update the frontier; --test finalizes
  benchmark.py            candidate evaluation jobs and the validation frontier
  inner_loop.py           one memory system on one dataset (prepare on train, evaluate)
  agents/                 seed harnesses (no_memory, fewshot_all) and their few-shot base class
  .claude/skills/         proposer instructions (SKILL.md)
  docker/proposer/        proposer image
  .env.example            credentials and run paths for one text-classification run
agentic/                  HotpotQA and AppWorld framework
  core/                   loop (meta_harness.py), evaluation, harness interface, LLM client and budgets, audit
  domains/hotpotqa/       task, retrieval tools, seed harness, proposer instructions, BM25 retriever server
  domains/appworld/       task, environment tools, seed harness (ReAct port), proposer instructions, custom splits
  configs/                common.yaml and one config per optimization setting
  scripts/                create_run.py, run_one.sh, preflight.sh, finalize.sh, appworld_data_setup.sh
  data/hotpotqa/          split IDs, sampling protocol, corpus and index builders (DATA.md)
  docker/proposer/        proposer image
  .env.example            credentials and endpoints read by run_one.sh
harnesses/                text-classification and HotpotQA candidate harnesses, by system (outputs, not inputs)
```

## Text classification

`text_classification/` is Meta-Harness `reference_examples/text_classification` (commit `44b9942`) with test
isolation, the proposer sandbox and the settings in `config.yaml`. Each run uses its own copy; the proposer container
mounts only `$RUN/checkout`, so results, test data and caches stay outside it:

```bash
REPO=$PWD; RUN=$PWD/runs/text-20b; mkdir -p $RUN/checkout/reference_examples $RUN/test-data
cp text_classification/.env.example text_classification/.env && chmod 600 text_classification/.env   # fill in the token
cp -r text_classification $RUN/checkout/reference_examples/text_classification
rm -f $RUN/checkout/reference_examples/text_classification/.env   # the proposer container mounts this copy
cd $RUN/checkout/reference_examples/text_classification && uv sync --frozen
docker build -t text-proposer:v1 docker/proposer
```

From the upstream commit, copy `data/{uspto,symptom_diagnosis,crime_prediction}/{train,val}.jsonl` into `data/` and
the matching `test.jsonl` files into `$RUN/test-data/<same folder>/`. Serve `gpt-oss-20b` at the `api_base` in
`config.yaml`, then:

```bash
set -a; . $REPO/text_classification/.env; set +a
uv run python meta_harness.py --run-name text-20b --iterations 20 --fresh   # optimization (validation only)
uv run python meta_harness.py --run-name text-20b --test                    # once, after optimization
```

The selected harness is the first entry of the validation Pareto frontier (equal-dataset macro accuracy versus
memory context length); test results never affect selection.

## HotpotQA and AppWorld

```bash
cd agentic && uv sync --frozen
docker build -t agentic-proposer:v1 -f docker/proposer/Dockerfile .
cp .env.example .env && chmod 600 .env    # then fill in CLAUDE_CODE_OAUTH_TOKEN
```

- HotpotQA: build the splits and the BM25 retriever as in `data/hotpotqa/DATA.md`, and serve `Qwen/Qwen3-8B` as
  `qwen3-8b` with vLLM (`--reasoning-parser qwen3 --max-model-len 16384`).
- AppWorld: run `scripts/appworld_data_setup.sh` (installs the separately distributed AppWorld package and
  data) and serve `gpt-oss-120b` with vLLM.

Serve the solver and the retriever on an address the container reaches through `host.docker.internal`, set
`AGENTIC_SOLVER_API_BASE` in `.env` (and `domain_config.retriever_url` for HotpotQA), then:

```bash
scripts/run_one.sh configs/hotpotqa-8b.yaml                              # run hotpotqa-8b-seed0 (20 iterations)
scripts/run_one.sh configs/hotpotqa-8b.yaml --run-id hotpotqa-8b-seed1   # another independent run
scripts/finalize.sh hotpotqa-8b-seed0                                    # once, after the loop has stopped
```

For an authenticated solver, set `AGENTIC_SOLVER_API_KEY` in `.env`. Saved host, proposer and evaluation configs
contain only `solver.api_key_env` (the environment variable name); the client reads its value at runtime. Docker
receives the variable by name, keeping its value out of the command line. An unset or empty default variable uses
the `local` placeholder for an unauthenticated solver. To use another variable, set `solver.api_key_env` to its name
and export it; use a dedicated solver variable, since `ANTHROPIC_API_KEY` is cleared for proposer subscription auth.
A missing custom variable is an error. Inline `solver.api_key` values are rejected, except for the old
empty/`local` placeholder. When calling `core.evaluate` directly, export the variable first.
Existing run checkouts contain copies of the framework and are not upgraded by updating this repository;
create a new run or migrate those copies and remove any previously saved inline keys before sharing them.

`run_one.sh` creates `runs/<run_id>/`, checks the setup and starts the loop in the background; running it again
resumes the run. Candidates are scored on the validation split during optimization. `finalize.sh` evaluates the
validation Pareto set, the top five by validation score and the validation best on the hidden selection split,
picks the best one (HotpotQA exact match, AppWorld task goal completion), and then evaluates the seed and these
candidates once on the test split, which is never used for selection.

## Citation

<!-- TODO: fill in the arXiv ID once the preprint is out. -->
```bibtex
@article{kim2026elicitation,
  title   = {Elicitation or Expansion? Demystifying Gains from Agent Harness Optimization},
  author  = {Kim, Younghyun and Seo, Wooseok and Kim, Minseo and Oh, Seunghyuk and
             Lee, Minjae and Kim, Donghoon and Koo, Hyung Il and Kim, Jaehyung and Kang, Wonjun},
  journal = {arXiv preprint arXiv:[ARXIV ID]},
  year    = {2026}
}
```

## License

[MIT](LICENSE). Third-party code and dataset terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
