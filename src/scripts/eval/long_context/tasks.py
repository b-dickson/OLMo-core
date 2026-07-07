"""
Synthetic long-context retrieval tasks: Needle-in-a-Haystack (NIAH) and a subset of
RULER-style synthetic tasks.

All tasks are *self-contained synthetic* (no external corpora to download): the
haystack is assembled from a fixed pool of neutral filler sentences, so a run is fully
reproducible from a seed. Each generator returns a list of :class:`Sample`.

Context length is controlled in *tokens* by the caller: each generator takes an
``approx_context_tokens`` target and a ``count_tokens`` callable, and pads the haystack
with filler until the tokenized prompt reaches the target (then the caller may trim).

References:
  - Needle-in-a-Haystack pressure test (Kamradt, 2023).
  - RULER (Hsieh et al., 2024), arXiv:2404.06654 -- task definitions adapted here.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# ── Filler pool ───────────────────────────────────────────────────────────────
# Neutral, low-information sentences. RULER permits noise filler in place of essays;
# we use that variant so the task needs no external download.
_FILLER_SENTENCES = [
    "The grass is green and the sky is blue.",
    "Water flows downhill and gathers in the valley.",
    "The morning light spread slowly across the quiet field.",
    "A gentle wind moved through the tall summer grass.",
    "The river continued its steady course toward the sea.",
    "Clouds drifted overhead without any particular hurry.",
    "The old wooden bridge creaked under the weight of the cart.",
    "Birds settled into the branches as the evening approached.",
    "The path wound between the hills and out of sight.",
    "Rain fell softly on the roofs of the sleeping town.",
    "The lamp on the desk gave off a warm and steady glow.",
    "Leaves turned over in the breeze, showing their pale undersides.",
    "The kettle sat quietly on the stove, not yet boiling.",
    "Snow covered the fields in a smooth, unbroken sheet.",
    "The harbor was calm, and the boats rocked only a little.",
]

# Pool of "keys" (words) and value generators for NIAH/RULER tasks.
_WORDS = [
    "elephant", "kingdom", "lantern", "harvest", "compass", "meadow", "anchor",
    "thunder", "marble", "willow", "cobalt", "saffron", "juniper", "quartz",
    "falcon", "ember", "glacier", "horizon", "lattice", "nectar", "obsidian",
    "petal", "ripple", "summit", "tundra", "velvet", "whisper", "zephyr",
]


def _rng_word(rng: random.Random) -> str:
    return rng.choice(_WORDS)


def _magic_number(rng: random.Random, digits: int = 7) -> str:
    return "".join(rng.choice(string.digits) for _ in range(digits))


@dataclass
class Sample:
    """A single long-context eval example."""

    task: str
    prompt: str
    answers: List[str]                 # acceptable answer string(s); scoring is recall
    approx_context_tokens: int
    depth_frac: Optional[float] = None  # for NIAH heatmap (where the needle sits)
    meta: Dict = field(default_factory=dict)


def _pad_to_tokens(
    body_prefix: str,
    needle_block: str,
    body_suffix: str,
    depth_frac: float,
    approx_context_tokens: int,
    count_tokens: Callable[[str], int],
    rng: random.Random,
) -> str:
    """
    Build a haystack of filler sentences targeting ``approx_context_tokens`` tokens,
    insert ``needle_block`` at fractional depth ``depth_frac``, and wrap with
    prefix/suffix (question framing). Filler is added until the token target is met.
    """
    # Grow filler list until we hit the token budget (measured on filler alone).
    filler: List[str] = []
    # Cheap geometric growth then fine top-up.
    while count_tokens(" ".join(filler)) < approx_context_tokens:
        filler.extend(rng.sample(_FILLER_SENTENCES, k=len(_FILLER_SENTENCES)))
    # Insert needle at depth.
    insert_at = int(len(filler) * depth_frac)
    filler = filler[:insert_at] + [needle_block] + filler[insert_at:]
    haystack = " ".join(filler)
    return f"{body_prefix}{haystack}\n\n{body_suffix}"


# ── NIAH: classic single needle, depth-swept (for the heatmap) ─────────────────

def niah_single(
    *,
    approx_context_tokens: int,
    count_tokens: Callable[[str], int],
    num_samples: int,
    depths: List[float],
    seed: int = 0,
) -> List[Sample]:
    """
    RULER niah_single_1 style: one magic number for one key, retrieve the number.
    Generates ``num_samples`` per depth in ``depths`` (for the depth x length heatmap).
    """
    rng = random.Random(seed)
    out: List[Sample] = []
    for depth in depths:
        for _ in range(num_samples):
            key = _rng_word(rng)
            val = _magic_number(rng)
            needle = f" One of the special magic numbers for {key} is: {val}. "
            prefix = (
                "Below is a long document. Read it carefully; a special magic number is "
                "hidden somewhere inside.\n\n"
            )
            suffix = (
                f"Question: What is the special magic number for {key}?\n"
                f"Answer: The special magic number for {key} is"
            )
            prompt = _pad_to_tokens(prefix, needle, suffix, depth,
                                    approx_context_tokens, count_tokens, rng)
            out.append(Sample(
                task="niah_single", prompt=prompt, answers=[val],
                approx_context_tokens=approx_context_tokens, depth_frac=depth,
                meta={"key": key},
            ))
    return out


# ── RULER: multi-key (distractor) NIAH ─────────────────────────────────────────

def niah_multikey(
    *,
    approx_context_tokens: int,
    count_tokens: Callable[[str], int],
    num_samples: int,
    num_keys: int = 4,
    seed: int = 0,
) -> List[Sample]:
    """Several keys each with a number; retrieve the one that is queried (distractors present)."""
    rng = random.Random(seed + 1)
    out: List[Sample] = []
    for _ in range(num_samples):
        keys = rng.sample(_WORDS, k=num_keys)
        vals = [_magic_number(rng) for _ in keys]
        needles = "".join(
            f" One of the special magic numbers for {k} is: {v}. " for k, v in zip(keys, vals)
        )
        target_idx = rng.randrange(num_keys)
        prefix = "Below is a long document with several special magic numbers hidden inside.\n\n"
        suffix = (
            f"Question: What is the special magic number for {keys[target_idx]}?\n"
            f"Answer: The special magic number for {keys[target_idx]} is"
        )
        prompt = _pad_to_tokens(prefix, needles, suffix, 0.5,
                                approx_context_tokens, count_tokens, rng)
        out.append(Sample(
            task="niah_multikey", prompt=prompt, answers=[vals[target_idx]],
            approx_context_tokens=approx_context_tokens, depth_frac=0.5,
            meta={"keys": keys, "target": keys[target_idx]},
        ))
    return out


# ── RULER: multi-value (one key, several values) ───────────────────────────────

def niah_multivalue(
    *,
    approx_context_tokens: int,
    count_tokens: Callable[[str], int],
    num_samples: int,
    num_values: int = 4,
    seed: int = 0,
) -> List[Sample]:
    """One key associated with several numbers scattered in the context; retrieve all."""
    rng = random.Random(seed + 2)
    out: List[Sample] = []
    for _ in range(num_samples):
        key = _rng_word(rng)
        vals = [_magic_number(rng) for _ in range(num_values)]
        needles = "".join(
            f" One of the special magic numbers for {key} is: {v}. " for v in vals
        )
        prefix = "Below is a long document; the same key has several magic numbers.\n\n"
        suffix = (
            f"Question: List all the special magic numbers for {key}.\n"
            f"Answer: The special magic numbers for {key} are"
        )
        prompt = _pad_to_tokens(prefix, needles, suffix, 0.5,
                                approx_context_tokens, count_tokens, rng)
        out.append(Sample(
            task="niah_multivalue", prompt=prompt, answers=vals,
            approx_context_tokens=approx_context_tokens, depth_frac=0.5,
            meta={"key": key, "n_values": num_values},
        ))
    return out


# ── RULER: variable tracking (multi-hop) ───────────────────────────────────────

def variable_tracking(
    *,
    approx_context_tokens: int,
    count_tokens: Callable[[str], int],
    num_samples: int,
    chain_len: int = 4,
    seed: int = 0,
) -> List[Sample]:
    """
    VT: VAR X = <number>; then VAR Y = VAR X; ... find all variables equal to the value.
    Tests multi-hop coreference across the context.
    """
    rng = random.Random(seed + 3)
    out: List[Sample] = []
    for _ in range(num_samples):
        value = _magic_number(rng)
        var_names = [f"VAR_{''.join(rng.choice(string.ascii_uppercase) for _ in range(3))}"
                     for _ in range(chain_len)]
        # Deduplicate.
        var_names = list(dict.fromkeys(var_names))
        stmts = [f" {var_names[0]} = {value}. "]
        for i in range(1, len(var_names)):
            stmts.append(f" {var_names[i]} = {var_names[i-1]}. ")
        needle = "".join(stmts)
        prefix = "Below is a long document containing variable assignments.\n\n"
        suffix = (
            f"Question: Find all variables that are assigned the value {value} "
            f"(directly or through a chain of assignments). List their names.\n"
            f"Answer: The variables equal to {value} are"
        )
        prompt = _pad_to_tokens(prefix, needle, suffix, 0.5,
                                approx_context_tokens, count_tokens, rng)
        out.append(Sample(
            task="variable_tracking", prompt=prompt, answers=var_names,
            approx_context_tokens=approx_context_tokens, depth_frac=0.5,
            meta={"value": value, "chain_len": len(var_names)},
        ))
    return out


# ── Registry ────────────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, Callable] = {
    "niah_single": niah_single,
    "niah_multikey": niah_multikey,
    "niah_multivalue": niah_multivalue,
    "variable_tracking": variable_tracking,
}


def score_sample(sample: Sample, generation: str) -> float:
    """
    Recall score in [0, 1]: fraction of expected answer strings that appear in the
    generated text. For single-answer tasks this is 0/1 substring match.
    """
    gen = generation.lower()
    hits = sum(1 for a in sample.answers if a.lower() in gen)
    return hits / max(1, len(sample.answers))
