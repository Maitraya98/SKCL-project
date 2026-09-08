# SKCL Reproduction & Domain-Shift Study

Reproduction of **SKCL** (Semantic Knowledge-driven Contrastive Learning), **BCL**
(Balanced Contrastive Learning), and **ConCutMix** on CIFAR-100/10-LT (β=100), plus
an original experiment measuring whether SKCL's LLM-derived semantic prototypes are
more robust to day/night domain shift than BCL's purely visual ones, using the
Oxford RobotCar / ROAD dataset.

> ⚠️ **This repository was reconstructed from Claude chat history, not copied from
> the original working tree.** See [`NOTES.md`](NOTES.md) for exactly what's
> verified-verbatim vs. reconstructed vs. missing entirely. Treat this as a strong
> starting point to diff against your real cluster copy (`~/skcl-project` on
> TinyGPU), not as a guaranteed-correct drop-in replacement.

## Project structure

Part 1 reproduces three methods on CIFAR-100-LT/CIFAR-10-LT (β=100):
- **BCL** — public code exists but only supports ImageNet scale; adapted here.
- **ConCutMix** — public code exists but needed compatibility fixes to run.
- **SKCL** — no public code; implemented from the paper's description.

Part 2 is a novel experiment: SKCL's text descriptions don't depend on lighting,
while BCL's purely visual features do. Both are trained on **daytime-only** driving
photos (ROAD/RobotCar) and evaluated on held-out day and night frames, to measure
whether SKCL's semantic anchoring gives it a smaller domain-shift accuracy drop.

```
skcl-project/
├── src/
│   ├── datasets/
│   │   ├── cifar_lt.py            # long-tailed CIFAR-10/100 sampler (shared by all 3 methods)
│   │   └── cifar_hierarchy.py     # fine/coarse class hierarchy for CIFAR
│   ├── llm_descriptions/
│   │   └── generate_descriptions.py   # LLM class-description generation (api/local/offline)
│   ├── semantic_graph/
│   │   └── build_graph.py         # SBERT embeddings + Top-K semantic similarity graph
│   └── robotcar/                  # [see NOTES.md — not recovered]
├── models/
│   ├── resnet_cifar.py            # ResNet-32 backbone + NormedLinear + BCLModelCIFAR
│   └── skcl_model.py              # two-branch SKCL model (fine+coarse classifiers, prototypes)
├── loss/
│   └── skcl_loss.py               # Eq. 4: prototype-alignment + class-balanced contrastive loss
├── tests/
│   └── test_cifar_lt.py           # synthetic sanity tests for the LT sampler
├── scripts/
│   └── inspect_cifar_lt.py        # run on cluster: verify real counts, save distribution figure
├── jobs/                          # SLURM (TinyGPU) batch scripts
└── docs/                          # report-writing notes, if any survive
```

## Method summary

- **SKCL**: an LLM writes a short description of each class; SBERT (`all-MiniLM-L6-v2`)
  embeds those descriptions; a Top-K cosine-similarity graph links each class to its
  semantic neighbors (fine or coarse). The model has two branches — a contrastive
  branch trained with `SKCLLoss` (Eq. 4: pull each sample toward its own class's
  prototype and its semantic neighbors, plus a class-balanced contrastive term) and a
  two-level (fine+coarse) classification branch.
- **BCL**: purely visual two-branch contrastive learning, no text.
- Both share the same CIFAR ResNet-32 backbone (`resnet_cifar.py`) so the comparison
  isolates what the text descriptions specifically add.

## Environment

```bash
conda create -n skcl python=3.10
conda activate skcl
pip install torch torchvision sentence-transformers numpy matplotlib
# --backend local in generate_descriptions.py additionally needs:
pip install transformers accelerate
```

On TinyGPU (FAU HPC), remember the proxy for anything that needs internet access
(HuggingFace downloads, pip installs):
```bash
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
```

## Running the pipeline (CIFAR-100-LT, β=100)

```bash
# 1. Generate LLM class descriptions
python3 src/llm_descriptions/generate_descriptions.py \
    --dataset cifar100 --data_root $WORK/data --backend local \
    --local_model Qwen/Qwen2.5-3B-Instruct \
    --out descriptions_cifar100_real.json

# 2. Build the semantic similarity graph (Top-K=2)
python3 src/semantic_graph/build_graph.py \
    --descriptions descriptions_cifar100_real.json \
    --dataset cifar100 --data_root $WORK/data \
    --top_k 2 --out graph_cifar100_real.json

# 3. Sanity-check the long-tailed sampler against real cached data
python3 scripts/inspect_cifar_lt.py --root $WORK/data --dataset cifar100 --imb_factor 0.01

# 4. Launch training jobs
sbatch.tinygpu jobs/train_bcl_cifar.sbatch
sbatch.tinygpu jobs/train_concutmix_cifar.sbatch
sbatch.tinygpu jobs/train_skcl_cifar.sbatch
```

Each CIFAR job takes roughly 30–40 minutes on an RTX 2080 Ti (200 epochs); the
RobotCar jobs are faster per-epoch but run more epochs with day/night eval baked in.

## Tests

```bash
python3 tests/test_cifar_lt.py     # instant, no data needed
python3 models/resnet_cifar.py     # shape/structure check
python3 models/skcl_model.py       # shape/structure check
python3 loss/skcl_loss.py          # synthetic forward+backward check
```

## License / attribution

BCL and ConCutMix baselines adapt public repositories (linked in `NOTES.md`) under
their original licenses. SKCL has no public reference implementation; the code here
is an independent implementation from the paper's method description.
