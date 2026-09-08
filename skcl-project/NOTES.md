# Reconstruction notes

This repo was rebuilt by searching and reading past Claude chat conversations
about this project, at the user's request, because the original working tree
lives only on the FAU TinyGPU cluster (`~/skcl-project`) and wasn't uploaded
directly. **Nothing here was copied from a live filesystem — it was reassembled
from quoted code fragments across many chat turns.** Treat it as a draft to
diff against the real cluster copy, not a verified source of truth.

## Confidence per file

**Recovered verbatim (high confidence — appeared as complete, unbroken code
blocks in chat history):**
- `src/datasets/cifar_hierarchy.py`
- `models/skcl_model.py`
- `loss/skcl_loss.py`
- `src/semantic_graph/build_graph.py`

**Reconstructed from multiple partial fragments (medium confidence — logic and
structure match what was described/tested in chat, but exact line-for-line
text wasn't always visible):**
- `src/datasets/cifar_lt.py` — the mixin pattern, docstring, and class-count
  API (`get_cls_num_list`, `classify_many_medium_few` thresholds) are from
  chat; `get_img_num_per_cls`'s exact exponential formula and
  `_gen_imbalanced_data`'s shuffle logic are standard-protocol reconstructions
  consistent with the described test behavior (endpoints, monotonicity, seed
  reproducibility).
- `models/resnet_cifar.py` — `NormedLinear` and `BCLModelCIFAR` are verbatim;
  the `BasicBlock`/`ResNet_Cifar` backbone itself is the standard, widely-used
  CIFAR ResNet-32 (3×5 BasicBlocks, matches the "31 conv + 1 FC = 32 layers"
  assertion that was verified in chat) rather than a recovered original.
- `src/llm_descriptions/generate_descriptions.py` — the prompt template, the
  "output only the description" fix, the 5-class people-disambiguation dict,
  and the three-backend design (api/local/offline) are verbatim/near-verbatim.
  The internals of `call_api`/`call_local`/`call_offline` and `main()`'s
  `--regenerate` handling are reconstructed to match the described CLI usage.
- `tests/test_cifar_lt.py`, `scripts/inspect_cifar_lt.py` — reconstructed to
  match the described 7-test suite and cluster-inspection workflow.
- `jobs/*.sbatch` — reconstructed from the `module add python` / `conda
  activate skcl` / proxy-export pattern seen in terminal transcripts, plus
  filenames referenced elsewhere (`train_bcl_cifar.sbatch`, etc.). Exact
  `--time`, `--partition`, and script CLI flags are best-guess, not verified.

## Known gaps — not recovered, referenced only by name/description

These were mentioned in the project's methodology table but their code was
never quoted in a recoverable form:

- `src/robotcar/road_annotations.py` — parses ROAD dataset JSON annotations,
  tube-based subsampling, 20px minimum size filter.
- `src/robotcar/extract_road_dataset.py` — decodes RobotCar videos, crops
  annotated objects, assigns whole traversals to day-train/day-test/night-test.
- `src/robotcar/demosaic_robotcar.py` — raw Bayer-pattern → RGB conversion.
- `src/robotcar/robotcar_lt.py`, `robotcar_multiview.py`, `robotcar_hierarchy.py`
  — RobotCar equivalents of the CIFAR dataset/multiview/hierarchy files.
- `src/robotcar/survey_road_dataset.py`, `analyze_box_sizes.py` — pre-extraction
  analysis scripts.
- `cifar_lt_multiview.py` — 3-view augmentation wrapper (1 CE view + 2
  contrastive views) referenced in the pipeline diagram but not quoted.
- `train_bcl_cifar.py`, `train_skcl_cifar.py`, `train_bcl_robotcar.py`,
  `train_skcl_robotcar.py` — the actual training loop scripts the `jobs/*.sbatch`
  files invoke. Only their *role* was described (LR warmup + cosine decay,
  per-epoch eval, checkpointing); no training-loop code was recoverable.
- `loss/contrastive.py` (BalSCL) and `loss/logitadjust.py` — BCL's loss
  functions, adapted from the public
  [FlamieZhu/Balanced-Contrastive-Learning](https://github.com/FlamieZhu/Balanced-Contrastive-Learning)
  repo per chat history, but the adapted version's diffs weren't quoted.
- **ConCutMix**: chat history says its public repo needed 5 compatibility
  fixes to run, but the fixes themselves weren't quoted — re-derive by cloning
  the original ConCutMix repo and re-running it against `cifar_lt.py`.
- The LaTeX/Word project report itself (`ch1_introduction.tex` through
  `ch6_discussion.tex`, or the `.docx` version) — extensive content exists in
  chat history but reconstructing the full report wasn't in scope of "make a
  GitHub repo for this project"; ask if you'd like that pulled in too.

## Suggested next step

Diff every file here against your real `~/skcl-project` on TinyGPU
(`scp -r rlvl167v@tinygpu:~/skcl-project /tmp/real-skcl-project` then `diff -r`)
before trusting this for your report numbers. Where they disagree, the cluster
copy is ground truth — this repo is a recovery aid, not a replacement for it.
