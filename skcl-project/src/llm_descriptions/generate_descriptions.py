"""
generate_descriptions.py -- LLM description generation using the paper's
exact Section 3.2 template and prompt. Three backends:
  --backend api      : real OpenAI-compatible API call (needs OPENAI_API_KEY)
  --backend local     : local open-source instruct model on this GPU (no key needed)
  --backend offline    : deterministic stub text, for pipeline testing only

Supports three datasets: cifar10, cifar100, robotcar.

Two fixes applied over the initial version, found from inspecting real
Qwen2.5-3B output on CIFAR-100:
  1. The model sometimes echoed the prompt/template into its response
     instead of just answering -- added an explicit "output only the
     description" instruction.
  2. CIFAR-100's "baby" class (part of the official "people" superclass,
     along with boy/girl/man/woman) means a human infant, but a 3B model
     with no dataset context defaulted to describing a baby ANIMAL. Added
     a disambiguation hint for exactly these 5 known-ambiguous classes.

NOTE ON RECONSTRUCTION: the prompt template, disambiguation dict, and
backend selection were recovered verbatim from chat history. The body of
`call_api`, `call_local`, `call_offline`, and `main()` are reconstructed
to match the described CLI (--regenerate flag, --sleep throttling, JSON
output) rather than copied verbatim -- re-verify against your cluster
copy before regenerating real descriptions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "datasets"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "robotcar"))
from cifar_hierarchy import get_hierarchy  # noqa: E402

PROMPT_TEMPLATE = (
    "Please use the Template y to generate a description for class {cls}, "
    "ensuring the description does not exceed 300 words.\n\n"
    'Template y: "A description of the class {cls}, {{based on its appearance '
    'characteristics}} and {{behavioral traits or functional features}}."'
    "{disambiguation}"
    "\n\nOutput ONLY the description itself -- do not repeat this prompt, "
    "the template text, or add any preamble/explanation."
)

# CIFAR-100's official "people" superclass (baby/boy/girl/man/woman) is the
# one place a small local model reliably confused "class name" with "animal
# name" -- these hints disambiguate exactly those five classes.
PEOPLE_CLASSES_DISAMBIGUATION = {
    "baby": " (Note: this refers to a human infant/baby person, not an animal.)",
    "boy": " (Note: this refers to a human boy/male child, not an animal.)",
    "girl": " (Note: this refers to a human girl/female child, not an animal.)",
    "man": " (Note: this refers to an adult human male person, not an animal.)",
    "woman": " (Note: this refers to an adult human female person, not an animal.)",
}


def truncate_to_word_limit(text: str, max_words: int = 300) -> str:
    """Paper's Section 3.2 explicitly imposes a 300-word cap for comparison fairness."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def build_prompt(cls: str) -> str:
    disamb = PEOPLE_CLASSES_DISAMBIGUATION.get(cls, "")
    return PROMPT_TEMPLATE.format(cls=cls, disambiguation=disamb)


def call_offline(cls: str) -> str:
    """Deterministic stub, no model/network needed -- for pipeline plumbing tests only."""
    return f"A description of the class {cls}, based on its typical appearance and behavior."


def call_api(cls: str, model: str = "gpt-4") -> str:
    import openai  # local import: only required for --backend api

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(cls)}],
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


_LOCAL_MODEL_CACHE = {}


def call_local(cls: str, model_name: str = "Qwen/Qwen2.5-3B-Instruct") -> str:
    """Runs a local HF instruct model -- no API key needed, works on cluster GPUs."""
    if model_name not in _LOCAL_MODEL_CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        _LOCAL_MODEL_CACHE[model_name] = (tok, mdl)
    tok, mdl = _LOCAL_MODEL_CACHE[model_name]

    messages = [{"role": "user", "content": build_prompt(cls)}]
    input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
    out = mdl.generate(input_ids, max_new_tokens=400, do_sample=False)
    text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


BACKENDS = {"api": call_api, "local": call_local, "offline": call_offline}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "robotcar"], required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--backend", choices=list(BACKENDS.keys()), default="offline")
    parser.add_argument("--local_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--api_model", default="gpt-4")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sleep", type=float, default=0.0, help="throttle between API calls")
    parser.add_argument("--regenerate", nargs="*", default=None,
                         help="only regenerate these specific class names (patches an existing --out file)")
    args = parser.parse_args()

    fine_names, coarse_names, _ = get_hierarchy(args.dataset, args.data_root)
    all_classes = list(fine_names) + list(coarse_names)

    descriptions = {}
    if args.regenerate and os.path.exists(args.out):
        with open(args.out) as f:
            descriptions = json.load(f)
        all_classes = args.regenerate

    call_fn = BACKENDS[args.backend]
    for i, cls in enumerate(all_classes):
        if args.backend == "local":
            desc = call_fn(cls, args.local_model)
        elif args.backend == "api":
            desc = call_fn(cls, args.api_model)
        else:
            desc = call_fn(cls)
        desc = truncate_to_word_limit(desc)
        descriptions[cls] = desc
        print(f"[{i + 1}/{len(all_classes)}] {cls}: {desc[:80]}...")
        if args.sleep:
            time.sleep(args.sleep)

    with open(args.out, "w") as f:
        json.dump(descriptions, f, indent=2)
    print(f"{len(descriptions)} descriptions saved to {args.out}")


if __name__ == "__main__":
    main()
