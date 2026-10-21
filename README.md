# Elicitation or Expansion? Demystifying Gains from Agent Harness Optimization

<p align="center">
  Younghyun Kim<sup>1*</sup>, Wooseok Seo<sup>2*</sup>, Minseo Kim<sup>3</sup>, Seunghyuk Oh<sup>1</sup>, Minjae Lee<sup>1</sup>,<br>
  Donghoon Kim<sup>1</sup>, Hyung Il Koo<sup>1</sup>, Jaehyung Kim<sup>2†</sup>, Wonjun Kang<sup>1†</sup>
</p>
<p align="center">
  <sup>1</sup>FuriosaAI, <sup>2</sup>Yonsei University, <sup>3</sup>University of California, Berkeley<br>
  <sup>*</sup>Equal contribution, <sup>†</sup>Co-corresponding authors
</p>

<p align="center">
  <a href="https://furiosa-air.github.io/elicitation-or-expansion"><img alt="Project" src="https://img.shields.io/static/v1?label=Project&message=Page&color=blue&logo=github-pages"></a>
  <a href="https://arxiv.org/abs/2610.27889"><img alt="arXiv" src="https://img.shields.io/static/v1?label=arXiv&message=2610.27889&color=b31b1b&logo=arxiv"></a>
  <a href="https://huggingface.co/datasets/furiosa-ai/elicitation-or-expansion-trajectories"><img alt="Dataset" src="https://img.shields.io/static/v1?label=Dataset&message=Hugging%20Face&color=yellow&logo=huggingface"></a>
</p>

This repository contains the code that runs the harness optimizations in the paper and the candidate harnesses
produced by the text-classification and HotpotQA runs. Generated AppWorld candidate snapshots are excluded because
their comments may contain protected benchmark material; the AppWorld optimization framework and public baseline
port are included. The rollouts are released separately as a
[Hugging Face dataset](https://huggingface.co/datasets/furiosa-ai/elicitation-or-expansion-trajectories).

The proposer (Claude Opus 5 in Claude Code, effort max, auto-memory disabled) runs in an unprivileged Docker container
that sees only the run's checkout; the solver is any OpenAI-compatible endpoint (vLLM). Each run is 20 iterations with
two candidates per iteration. Candidate snapshots are in `harnesses/`; `seedN` identifies an independent optimization run.
The text-classification rollout baseline is `harnesses/text_classification/baseline/fewshot_all_reroll.py`.

| Task                | System          | Configuration                                 | Selected harnesses                                                                                                         |
| ------------------- | --------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Text classification | `20b-medium`  | `text_classification/config.yaml`           | `worked_case_memory.py`                                                                                                  |
| HotpotQA            | `8b`          | `agentic/configs/hotpotqa-8b.yaml`          | `linked_article_evidence.py` (seed0)`graded_evidence_detail.py` (seed1)`thinking_doubt_router.py` (seed2)            |
| HotpotQA            | `8b-relaxed`  | `agentic/configs/hotpotqa-8b-relaxed.yaml`  | `shared_attribute_intersection.py` (seed0)`residual_query_probe.py` (seed1)`shared_property_intersection.py` (seed2) |
| AppWorld            | `120b-medium` | `agentic/configs/appworld-120b-medium.yaml` | Not included (seed0, seed1, seed2)                                                                                         |

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
cp .env.example .env && chmod 600 .env
```

- **HotpotQA:** follow the [data setup](agentic/data/hotpotqa/DATA.md) and serve `Qwen/Qwen3-8B` as `qwen3-8b`
  with vLLM (`--reasoning-parser qwen3 --max-model-len 16384`).
- **AppWorld:** run `scripts/appworld_data_setup.sh` and serve `gpt-oss-120b` with vLLM.

Set `CLAUDE_CODE_OAUTH_TOKEN` and `AGENTIC_SOLVER_API_BASE` in `.env`, plus `AGENTIC_SOLVER_API_KEY` if needed.
For HotpotQA, also set `domain_config.retriever_url`. The solver and retriever must be reachable from Docker
through `host.docker.internal`.

Choose a configuration from the table above; for example:

```bash
scripts/run_one.sh configs/hotpotqa-8b.yaml
scripts/finalize.sh hotpotqa-8b-seed0    # after optimization finishes
```

`run_one.sh` starts or resumes optimization in the background; use `--run-id hotpotqa-8b-seed1` for another run.
`finalize.sh` selects a harness on the separate selection split, then evaluates on held-out test data.
Test results do not affect selection.

## Citation

```bibtex
@article{kim2026elicitation,
  title   = {Elicitation or Expansion? Demystifying Gains from Agent Harness Optimization},
  author  = {Kim, Younghyun and Seo, Wooseok and Kim, Minseo and Oh, Seunghyuk and
             Lee, Minjae and Kim, Donghoon and Koo, Hyung Il and Kim, Jaehyung and Kang, Wonjun},
  journal = {arXiv preprint arXiv:2610.27889},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.27889}
}
```

## License

[MIT](LICENSE). The files and components listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
remain under their own licenses. Dataset terms are also listed there.
