
#!/usr/bin/env python3
"""
CONTROLLED INSERTION INTO THE PILE
==================================

Parameter 1: Frequency
    - Explicit document IDs
    - Random documents
    - Every-Nth document
    - Selected ranges

Parameter 2: Dataset position
    - Early / middle / late / all
    - Custom dataset regions

Parameter 3: Within-document position
    - Beginning / middle / end
    - Character offset / fraction / random
    - Token offset / fraction / random
    - Character fraction + delimiter
    - Token fraction + delimiter

Additional functionality:
    - Any compatible Hugging Face tokenizer
    - Arbitrary literal or external-dataset spans
    - Random external passage sampling
    - Multiple independent targets
    - Per-document position overrides
    - Final token-position observations

Outputs:
    modified_dataset.jsonl
    original_hosts.jsonl
    insertion_metadata.json

Install:
    pip install -U datasets transformers

Run:
    python controlled_insertion.py
"""

import copy
import hashlib
import json
import math
import random

from collections import defaultdict
from itertools import islice
from pathlib import Path


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

CONFIG = {
    "pile": {
        "name": "EleutherAI/the_pile_deduplicated",
        "config": None,
        "split": "train",
        "revision": None,

        "take": {
            "mode": "first_n",
            "n": 100
        }

        # Alternatives:
        #
        # "take": {
        #     "mode": "source_ids",
        #     "ids": [2, 5, 100]
        # }
        #
        # "take": {
        #     "mode": "source_range",
        #     "start": 100,
        #     "stop": 200
        # }
    },

    # Only the entry point loads the tokenizer.
    # All insertion functions receive the object.
    "tokenizer_name": "gpt2",
    "tokenizer_revision": None,

    "seed": 42,
    "separator": "\n\n",
    "output_dir": "pile_run_001",

    # Manually configured target spans.
    "targets": [
        {
            "id": "example",

            "source": {
                "mode": "literal",
                "text": "my name is apratim"
            },

            "selection": {
                "mode": "explicit",
                "source_ids": [2, 5, 10]
            },

            "position": {
                "mode": "token_fraction_delimiter",
                "fraction": 0.5,
                "delimiters": [".", "\n"],
                "on_missing": "error"
            },

            "per_document_positions": {
                "2": {"mode": "beginning"},
                "5": {"mode": "end"}
            }
        }
    ],

    # Optional external dataset sampling.
    #
    # Set "targets": [] above and replace None with:
    #
    # {
    #     "dataset": "oskarvanderwal/bbq",
    #     "config": "All",
    #     "split": "test",
    #     "revision": None,
    #     "fields": ["context", "question"],
    #     "join_with": "\n",
    #     "count": 20,
    #     "pool_size": 500,
    #
    #     "extract": {
    #         "mode": "random_characters",
    #         "length": 100
    #     },
    #
    #     "selection": {
    #         "mode": "random",
    #         "frequency": 5,
    #         "stage": "all"
    #     },
    #
    #     "position": {
    #         "mode": "token_fraction",
    #         "fraction": 0.5
    #     },
    #
    #     "per_target": {
    #         "0": {
    #             "selection": {
    #                 "mode": "explicit",
    #                 "source_ids": [2, 5]
    #             }
    #         }
    #     }
    # }

    "sample_external": None
}


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def whole(value, label, minimum=0):
    """Require an integer >= minimum."""

    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{label} must be an integer >= {minimum}"
        )

    return value


def frac(value, label):
    """Require a finite fraction in [0, 1]."""

    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(
            f"{label} must be a finite fraction in [0, 1]"
        )

    return float(value)


def sha(value):
    """SHA-256 hash of UTF-8 encoded text."""

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


# ============================================================
# 3. HUGGING FACE DATASET STREAMING
# ============================================================

def stream_dataset(
    name,
    split="train",
    config=None,
    revision=None
):
    """
    Open a Hugging Face dataset as an ordered stream.

    No explicit shuffling is performed.
    """

    from datasets import load_dataset

    options = {
        "path": name,
        "split": split,
        "streaming": True
    }

    if config is not None:
        options["name"] = config

    if revision is not None:
        options["revision"] = revision

    return iter(load_dataset(**options))


# ============================================================
# 4. LOAD ORIGINAL PILE DOCUMENTS
# ============================================================

def source_ids_for_take(take):
    """
    Determine which original Pile source IDs to load.

    Modes:
        first_n
        source_ids
        source_range
    """

    mode = take["mode"]

    if mode == "first_n":

        ids = list(
            range(
                whole(take["n"], "take.n", 1)
            )
        )

    elif mode == "source_ids":

        ids = [
            whole(i, "source ID")
            for i in take["ids"]
        ]

    elif mode == "source_range":

        start = whole(
            take["start"],
            "start"
        )

        stop = whole(
            take["stop"],
            "stop",
            start + 1
        )

        ids = list(range(start, stop))

    else:
        raise ValueError(
            f"Unknown take mode: {mode}"
        )

    if not ids or len(set(ids)) != len(ids):
        raise ValueError(
            "Source IDs must be nonempty and unique"
        )

    return sorted(ids)


def load_hosts(pile):
    """
    Load existing Pile documents.

    Source IDs refer to original zero-based stream indices.
    Original ordering is preserved.
    """

    ids = source_ids_for_take(
        pile["take"]
    )

    desired = set(ids)

    rows = stream_dataset(
        pile["name"],
        pile.get("split", "train"),
        pile.get("config"),
        pile.get("revision")
    )

    hosts = []

    for idx, row in enumerate(
        islice(rows, ids[-1] + 1)
    ):

        if idx not in desired:
            continue

        value = row.get("text")

        if not isinstance(value, str):
            raise ValueError(
                f"Host {idx} has no string text field"
            )

        hosts.append({
            "source_id": idx,
            "text": value
        })

    if len(hosts) != len(ids):
        raise ValueError(
            "Some requested source documents "
            "were unavailable"
        )

    return hosts


# ============================================================
# 5. EXTERNAL DATASET TEXT EXTRACTION
# ============================================================

def read_fields(
    row,
    fields,
    join_with="\n"
):
    """
    Extract one or multiple fields from an external row.

    Examples:
        ["text"]
        ["context", "question"]
    """

    if isinstance(fields, str):
        fields = [fields]

    if (
        not isinstance(fields, list)
        or not fields
        or not isinstance(join_with, str)
    ):
        raise ValueError(
            "Invalid fields or join_with"
        )

    values = [
        row.get(field)
        for field in fields
    ]

    if any(
        not isinstance(v, str) or not v.strip()
        for v in values
    ):
        raise ValueError(
            f"Missing or empty string field in {fields}"
        )

    return join_with.join(values)


# ============================================================
# 6. MODEL-INDEPENDENT TOKENIZER INTERFACE
# ============================================================

def token_map(text, tokenizer):
    """
    Tokenize with the caller-supplied tokenizer.

    Compatible tokenizer requirements:
        - is_fast == True
        - Supports return_offsets_mapping
        - Returns character offsets

    Returns:
        offsets:
            List of (start_char, end_char)

        boundaries:
            N+1 potential insertion boundaries
            for N original tokens

    Boundary policy:
        - Adjacent offsets: after preceding token
        - Whitespace gap: after preceding token
        - Overlapping offsets: boundary unavailable

    Example:
        text = "Hello world."

        offsets = [
            (0, 5),
            (6, 11),
            (11, 12)
        ]

        boundaries = [
            0, 5, 11, 12
        ]

    Empty text:
        offsets = []
        boundaries = [0]
    """

    if (
        tokenizer is None
        or not getattr(tokenizer, "is_fast", False)
    ):
        raise ValueError(
            "Token operations require a fast tokenizer "
            "passed as an argument"
        )

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False
    )

    offsets = [
        tuple(pair)
        for pair in encoded["offset_mapping"]
    ]

    if not offsets:

        if text and not text.isspace():
            raise ValueError(
                "Tokenizer returned no offsets "
                "for non-whitespace text"
            )

        return [], [0]

    if any(
        start < 0
        or end < start
        or end > len(text)
        for start, end in offsets
    ):
        raise ValueError(
            "Tokenizer returned invalid offsets"
        )

    boundaries = [0]

    for left, right in zip(
        offsets,
        offsets[1:]
    ):

        previous_end = left[1]
        next_start = right[0]

        if previous_end <= next_start:

            # Allow gaps between token offsets.
            boundaries.append(previous_end)

        else:

            # Overlapping token offsets.
            boundaries.append(None)

    boundaries.append(len(text))

    return offsets, boundaries


# ============================================================
# 7. EXTRACT A PASSAGE FROM SOURCE TEXT
# ============================================================

def extract_passage(
    text,
    specification,
    tokenizer,
    rng
):
    """
    Passage extraction modes:

        all
        characters
        random_characters
        tokens
        random_tokens

    Token passage lengths are based on the original
    source document's tokenization.
    """

    if not isinstance(text, str) or not text:
        raise ValueError(
            "Source text is empty"
        )

    specification = (
        specification or {"mode": "all"}
    )

    mode = specification.get(
        "mode", "all"
    )

    if mode == "all":

        start = 0
        end = len(text)

    elif mode in (
        "characters",
        "random_characters"
    ):

        length = whole(
            specification["length"],
            "character passage length",
            1
        )

        if length > len(text):
            raise ValueError(
                "Source text is shorter "
                "than requested passage"
            )

        if mode == "characters":

            start = whole(
                specification["start"],
                "passage start"
            )

        else:

            start = rng.randrange(
                len(text) - length + 1
            )

        end = start + length

    elif mode in (
        "tokens",
        "random_tokens"
    ):

        length = whole(
            specification["length"],
            "token passage length",
            1
        )

        offsets, boundaries = token_map(
            text,
            tokenizer
        )

        if length > len(offsets):
            raise ValueError(
                "Source has insufficient tokens"
            )

        valid_starts = [
            i
            for i in range(
                len(offsets) - length + 1
            )
            if (
                boundaries[i] is not None
                and boundaries[i + length] is not None
                and boundaries[i] < boundaries[i + length]
            )
        ]

        if not valid_starts:
            raise ValueError(
                "No eligible source token passage"
            )

        if mode == "tokens":

            token_index = whole(
                specification["start"],
                "token passage start"
            )

        else:

            token_index = rng.choice(
                valid_starts
            )

        if token_index not in valid_starts:
            raise ValueError(
                "Requested token passage has "
                "an unavailable boundary"
            )

        start = boundaries[token_index]
        end = boundaries[token_index + length]

    else:
        raise ValueError(
            f"Unknown passage extraction mode: {mode}"
        )

    if not 0 <= start < end <= len(text):
        raise ValueError(
            "Invalid passage extraction range"
        )

    return text[start:end], {
        "mode": mode,
        "start_char": start,
        "end_char": end
    }


# ============================================================
# 8. LOAD AN INDIVIDUAL TARGET SPAN
# ============================================================

def get_span(
    source,
    tokenizer,
    rng
):
    """
    Source modes:
        literal
        hf

    The source may optionally contain an extract
    configuration specifying an arbitrary passage.
    """

    mode = source["mode"]

    if mode == "literal":

        full = source["text"]

        provenance = {
            "mode": "literal"
        }

    elif mode == "hf":

        row_id = whole(
            source["index"],
            "external row index"
        )

        rows = stream_dataset(
            source["dataset"],
            source.get("split", "train"),
            source.get("config"),
            source.get("revision")
        )

        row = next(
            islice(rows, row_id, row_id + 1),
            None
        )

        if row is None:
            raise ValueError(
                f"External row {row_id} not found"
            )

        full = read_fields(
            row,
            source.get("fields", "text"),
            source.get("join_with", "\n")
        )

        provenance = {
            "mode": "hf",
            "dataset": source["dataset"],
            "config": source.get("config"),
            "revision": source.get("revision"),
            "split": source.get(
                "split", "train"
            ),
            "row": row_id,
            "fields": source.get(
                "fields", "text"
            )
        }

    else:
        raise ValueError(
            f"Unknown target source mode: {mode}"
        )

    span, extraction = extract_passage(
        full,
        source.get("extract"),
        tokenizer,
        rng
    )

    provenance.update({
        "extraction": extraction,
        "sha256": sha(span)
    })

    return span, provenance


# ============================================================
# 9. AUTOMATIC EXTERNAL TARGET SAMPLING
# ============================================================

def sample_targets(
    specification,
    tokenizer,
    rng
):
    """
    Sample multiple unique passages from an external dataset.

    per_target allows individual overrides of:
        selection
        position
        per_document_positions
    """

    count = whole(
        specification["count"],
        "sample count",
        1
    )

    pool = whole(
        specification["pool_size"],
        "pool size",
        count
    )

    rows = list(
        islice(
            stream_dataset(
                specification["dataset"],
                specification.get(
                    "split", "train"
                ),
                specification.get("config"),
                specification.get("revision")
            ),
            pool
        )
    )

    indices = list(range(len(rows)))
    rng.shuffle(indices)

    targets = []
    seen = set()

    for row_id in indices:

        try:

            full = read_fields(
                rows[row_id],
                specification.get(
                    "fields", "text"
                ),
                specification.get(
                    "join_with", "\n"
                )
            )

            passage, extraction = extract_passage(
                full,
                specification.get("extract"),
                tokenizer,
                rng
            )

        except ValueError:
            continue

        if passage in seen:
            continue

        seen.add(passage)

        sequence = len(targets)

        overrides = specification.get(
            "per_target", {}
        ).get(
            str(sequence), {}
        )

        targets.append({
            "id": f"sample_{sequence}_{row_id}",

            "source": {
                "mode": "literal",
                "text": passage
            },

            "provenance": {
                "dataset": specification["dataset"],
                "config": specification.get("config"),
                "revision": specification.get("revision"),
                "split": specification.get(
                    "split", "train"
                ),
                "row": row_id,
                "fields": specification.get(
                    "fields", "text"
                ),
                "extraction": extraction,
                "sha256": sha(passage)
            },

            "selection": copy.deepcopy(
                overrides.get(
                    "selection",
                    specification["selection"]
                )
            ),

            "position": copy.deepcopy(
                overrides.get(
                    "position",
                    specification["position"]
                )
            ),

            "per_document_positions": copy.deepcopy(
                overrides.get(
                    "per_document_positions",
                    specification.get(
                        "per_document_positions", {}
                    )
                )
            )
        })

        if len(targets) == count:
            break

    if len(targets) != count:
        raise ValueError(
            f"Only {len(targets)} unique usable "
            f"passages found; requested {count}"
        )

    return targets


# ============================================================
# 10. DATASET REGION SELECTION
# ============================================================

def region(
    selection,
    total
):
    """
    Select a region within the loaded document pool.

    Stages:
        all
        early
        middle
        late
        custom
    """

    stage = selection.get(
        "stage", "all"
    )

    bounds = {
        "all": (0, total),

        "early": (
            0,
            total // 3
        ),

        "middle": (
            total // 3,
            2 * total // 3
        ),

        "late": (
            2 * total // 3,
            total
        )
    }

    if stage == "custom":

        start = int(
            total * frac(
                selection["start_fraction"],
                "start_fraction"
            )
        )

        end = int(
            total * frac(
                selection["end_fraction"],
                "end_fraction"
            )
        )

    elif stage in bounds:

        start, end = bounds[stage]

    else:
        raise ValueError(
            f"Unknown dataset stage: {stage}"
        )

    if not 0 <= start < end <= total:
        raise ValueError(
            "Dataset region is invalid or empty"
        )

    return list(range(start, end))


# ============================================================
# 11. SELECT HOST DOCUMENTS
# ============================================================

def choose_hosts(
    selection,
    hosts,
    span,
    rng
):
    """
    Host selection modes:
        explicit
        random
        range
        every_n

    Frequency refers to distinct scheduled host documents.

    Explicit IDs use original dataset source IDs.
    Other modes use loaded output indices.
    """

    total = len(hosts)
    mode = selection["mode"]

    lookup = {
        host["source_id"]: i
        for i, host in enumerate(hosts)
    }

    if mode == "explicit":

        source_ids = [
            whole(x, "source ID")
            for x in selection["source_ids"]
        ]

        if (
            len(source_ids) != len(set(source_ids))
            or any(
                x not in lookup
                for x in source_ids
            )
        ):
            raise ValueError(
                "Explicit IDs must be distinct and loaded"
            )

        selected = [
            lookup[x]
            for x in source_ids
        ]

    else:

        frequency = whole(
            selection["frequency"],
            "frequency"
        )

        if mode == "random":

            candidates = region(
                selection,
                total
            )

        elif mode == "range":

            start = whole(
                selection["start_output_index"],
                "range start"
            )

            end = whole(
                selection["stop_output_index"],
                "range stop",
                start + 1
            )

            if end > total:
                raise ValueError(
                    "Range exceeds loaded documents"
                )

            candidates = list(
                range(start, end)
            )

        elif mode == "every_n":

            start = whole(
                selection.get(
                    "start_output_index", 0
                ),
                "every_n start"
            )

            step = whole(
                selection["every_n"],
                "every_n",
                1
            )

            if start >= total:
                raise ValueError(
                    "Every-N start exceeds dataset length"
                )

            candidates = list(
                range(start, total, step)
            )

        else:
            raise ValueError(
                f"Unknown selection mode: {mode}"
            )

        if mode == "every_n":

            if frequency > len(candidates):
                raise ValueError(
                    "Not enough documents at "
                    "the requested interval"
                )

            selected = candidates[:frequency]

        else:

            eligible = [
                i
                for i in candidates
                if span not in hosts[i]["text"]
            ]

            if frequency > len(eligible):
                raise ValueError(
                    "Insufficient eligible host documents"
                )

            selected = rng.sample(
                eligible,
                frequency
            )

    # An existing occurrence is not counted as a new insertion.
    if any(
        span in hosts[i]["text"]
        for i in selected
    ):
        raise ValueError(
            "Selected host already contains target text"
        )

    return sorted(selected)


# ============================================================
# 12. POSITION WITHIN ORIGINAL DOCUMENT
# ============================================================

def position_offset(
    text,
    specification,
    tokenizer,
    rng
):
    """
    Resolve position to a character offset in ORIGINAL text.

    Character modes:
        beginning
        middle
        end
        character_offset
        character_fraction
        percentage
        random_character

    Token modes:
        token_offset
        token_fraction
        token_percentage
        random_token

    Delimiter modes:
        character_fraction_delimiter
        token_fraction_delimiter

    The tokenizer is always supplied by the caller.
    """

    mode = specification["mode"]
    text_length = len(text)

    info = {
        "mode": mode
    }

    boundaries = None

    # --------------------------------------------------------
    # CHARACTER-BASED POSITIONS
    # --------------------------------------------------------

    if mode == "beginning":

        offset = 0

    elif mode == "middle":

        offset = text_length // 2

    elif mode == "end":

        offset = text_length

    elif mode == "character_offset":

        offset = whole(
            specification["value"],
            "character offset"
        )

    elif mode == "random_character":

        offset = rng.randint(
            0, text_length
        )

    elif mode in (
        "character_fraction",
        "percentage",
        "character_fraction_delimiter"
    ):

        value = frac(
            specification.get(
                "fraction",
                specification.get("value")
            ),
            "character fraction"
        )

        offset = round(
            text_length * value
        )

        info["requested_fraction"] = value

    # --------------------------------------------------------
    # TOKEN-BASED POSITIONS
    # --------------------------------------------------------

    elif mode in (
        "token_offset",
        "token_fraction",
        "token_percentage",
        "random_token",
        "token_fraction_delimiter"
    ):

        offsets, boundaries = token_map(
            text,
            tokenizer
        )

        token_count = len(offsets)

        if mode == "token_offset":

            token_index = whole(
                specification["value"],
                "token offset"
            )

        elif mode == "random_token":

            available = [
                i
                for i, boundary in enumerate(boundaries)
                if boundary is not None
            ]

            token_index = rng.choice(
                available
            )

        else:

            value = frac(
                specification.get(
                    "fraction",
                    specification.get("value")
                ),
                "token fraction"
            )

            token_index = round(
                token_count * value
            )

            info["requested_fraction"] = value

        if token_index > token_count:
            raise ValueError(
                f"Token boundary {token_index} "
                f"exceeds {token_count} tokens"
            )

        info.update({
            "original_token_count": token_count,
            "requested_token_boundary": token_index
        })

        if boundaries[token_index] is None:

            if mode != "token_fraction_delimiter":
                raise ValueError(
                    f"Token boundary {token_index} "
                    "overlaps; exact insertion unavailable"
                )

            # Delimiter alignment may search forward
            # even when the requested boundary overlaps.
            offset = offsets[token_index - 1][1]

            info["requested_boundary_overlaps"] = True

        else:

            offset = boundaries[token_index]

            info["requested_boundary_overlaps"] = False

    else:
        raise ValueError(
            f"Unknown insertion position mode: {mode}"
        )

    if offset > text_length:
        raise ValueError(
            "Insertion position exceeds document length"
        )

    info["requested_character_offset"] = offset

    # --------------------------------------------------------
    # DELIMITER ALIGNMENT
    # --------------------------------------------------------

    if mode.endswith("_delimiter"):

        delimiters = specification.get(
            "delimiters"
        )

        if (
            not isinstance(delimiters, list)
            or not delimiters
            or any(
                not isinstance(d, str) or not d
                for d in delimiters
            )
        ):
            raise ValueError(
                "delimiters must be a nonempty list "
                "of nonempty strings"
            )

        # Character mode allows any matching delimiter.
        # Token mode requires its end to be a valid
        # original-token boundary.
        allowed_boundaries = (
            None
            if boundaries is None
            else {
                boundary
                for boundary in boundaries
                if boundary is not None
            }
        )

        matches = []

        for order, delimiter in enumerate(delimiters):

            start = text.find(
                delimiter,
                offset
            )

            while start >= 0:

                delimiter_end = (
                    start + len(delimiter)
                )

                if (
                    allowed_boundaries is None
                    or delimiter_end in allowed_boundaries
                ):

                    matches.append((
                        start,
                        order,
                        delimiter_end,
                        delimiter
                    ))

                    break

                start = text.find(
                    delimiter,
                    start + 1
                )

        if matches:

            # Choose the earliest starting delimiter.
            # List order breaks ties.
            (
                found,
                _,
                offset,
                matched_delimiter
            ) = min(matches)

            info.update({
                "matched_delimiter": matched_delimiter,
                "delimiter_start": found
            })

        else:

            policy = specification.get(
                "on_missing", "error"
            )

            if policy == "end":

                offset = text_length

            elif policy == "requested":

                if info.get(
                    "requested_boundary_overlaps",
                    False
                ):
                    raise ValueError(
                        "Cannot fall back to an "
                        "overlapping token boundary"
                    )

                offset = info[
                    "requested_character_offset"
                ]

            elif policy == "error":

                raise ValueError(
                    "No eligible delimiter found after "
                    "requested position"
                )

            else:
                raise ValueError(
                    "Invalid on_missing policy"
                )

            info["delimiter_fallback"] = policy

    info[
        "actual_original_character_offset"
    ] = offset

    if boundaries is not None:

        indices = [
            i
            for i, boundary in enumerate(boundaries)
            if boundary == offset
        ]

        actual_index = (
            indices[0]
            if indices
            else None
        )

        info["actual_original_token_boundary"] = (
            actual_index
        )

        info["actual_original_token_fraction"] = (
            actual_index / (len(boundaries) - 1)
            if actual_index is not None
            and len(boundaries) > 1
            else None
        )

    return offset, info


# ============================================================
# 13. FINAL MODIFIED TOKEN POSITION
# ============================================================

def final_token_observation(
    modified_text,
    records,
    tokenizer
):
    """
    Tokenize each modified document only once.

    For every inserted span, record:
        - Modified token count
        - Token interval overlapping the inserted characters
        - First overlapping token fraction
        - Tokens crossing span boundaries

    Token intervals use [start, end).
    """

    if tokenizer is None:
        return

    offsets, _ = token_map(
        modified_text,
        tokenizer
    )

    token_count = len(offsets)

    for record in records:

        span_start = record[
            "inserted_start_char"
        ]

        span_end = record[
            "inserted_end_char"
        ]

        overlapping = [
            i
            for i, (start, end) in enumerate(offsets)
            if start < span_end and end > span_start
        ]

        crossing = [
            i
            for i in overlapping
            if (
                offsets[i][0] < span_start
                or offsets[i][1] > span_end
            )
        ]

        record["final_token_observation"] = {
            "modified_token_count": token_count,

            "span_overlapping_token_interval": (
                [
                    overlapping[0],
                    overlapping[-1] + 1
                ]
                if overlapping
                else None
            ),

            "span_first_token_fraction": (
                overlapping[0] / token_count
                if overlapping
                else None
            ),

            "boundary_crossing_token_indices": crossing,

            "no_overlapping_tokens": not overlapping
        }


# ============================================================
# 14. CONTROLLED INSERTION ENGINE
# ============================================================

def insert_targets(
    config,
    hosts,
    tokenizer=None
):
    """
    Insert all configured targets into existing documents.

    Important:
        All insertion positions use ORIGINAL document text.

        Multiple targets can share a host and can even
        share the same original insertion position.

        The tokenizer is supplied by the caller.

    Returns:
        output_documents
        target_information
        insertion_records
    """

    rng = random.Random(
        config.get("seed", 42)
    )

    source_ids = [
        host["source_id"]
        for host in hosts
    ]

    if (
        not hosts
        or source_ids != sorted(set(source_ids))
        or any(
            not isinstance(host["text"], str)
            for host in hosts
        )
    ):
        raise ValueError(
            "Hosts must be sorted, unique, "
            "and contain string text"
        )

    targets = copy.deepcopy(
        config.get("targets", [])
    )

    # Add automatically sampled external spans.
    if config.get("sample_external"):

        targets.extend(
            sample_targets(
                config["sample_external"],
                tokenizer,
                rng
            )
        )

    target_ids = [
        target["id"]
        for target in targets
    ]

    if len(set(target_ids)) != len(target_ids):
        raise ValueError(
            "Target IDs must be unique"
        )

    planned = defaultdict(list)
    target_information = []
    seen_spans = []

    # --------------------------------------------------------
    # A. PLAN ALL INSERTIONS
    # --------------------------------------------------------

    for order, target in enumerate(targets):

        span, provenance = get_span(
            target["source"],
            tokenizer,
            rng
        )

        # Avoid direct containment between target strings.
        if any(
            span in other or other in span
            for other in seen_spans
        ):
            raise ValueError(
                "Configured target strings overlap "
                "by substring containment"
            )

        seen_spans.append(span)

        selected_hosts = choose_hosts(
            target["selection"],
            hosts,
            span,
            rng
        )

        for index in selected_hosts:

            host = hosts[index]

            position = target.get(
                "per_document_positions", {}
            ).get(
                str(host["source_id"]),
                target["position"]
            )

            offset, details = position_offset(
                host["text"],
                position,
                tokenizer,
                rng
            )

            planned[index].append({
                "id": target["id"],
                "span": span,
                "offset": offset,
                "order": order,
                "position": details
            })

        target_information.append({
            "id": target["id"],
            "span": span,
            "span_sha256": sha(span),

            "source": target.get(
                "provenance",
                provenance
            ),

            "frequency": len(selected_hosts),

            "host_source_ids": [
                hosts[index]["source_id"]
                for index in selected_hosts
            ]
        })

    separator = config.get(
        "separator",
        "\n\n"
    )

    if not isinstance(separator, str):
        raise ValueError(
            "separator must be a string"
        )

    output_documents = []
    insertion_records = []

    # --------------------------------------------------------
    # B. APPLY INSERTIONS TO ORIGINAL DOCUMENTS
    # --------------------------------------------------------

    for index, host in enumerate(hosts):

        original = host["text"]

        placements = sorted(
            planned[index],
            key=lambda item: (
                item["offset"],
                item["order"]
            )
        )

        pieces = []
        own_records = []

        cursor = 0
        emitted = 0

        for placement in placements:

            offset = placement["offset"]

            # Preserve untouched original text.
            segment = original[
                cursor:offset
            ]

            pieces.append(segment)
            emitted += len(segment)

            left_separator = (
                separator
                if offset > 0
                else ""
            )

            right_separator = (
                separator
                if offset < len(original)
                else ""
            )

            span_start = (
                emitted + len(left_separator)
            )

            span_end = (
                span_start + len(
                    placement["span"]
                )
            )

            pieces.extend((
                left_separator,
                placement["span"],
                right_separator
            ))

            emitted += (
                len(left_separator)
                + len(placement["span"])
                + len(right_separator)
            )

            own_records.append({
                "target_id": placement["id"],

                "pile_source_id": host["source_id"],
                "output_document_index": index,

                "original_character_offset": offset,

                "inserted_start_char": span_start,
                "inserted_end_char": span_end,

                "left_separator": left_separator,
                "right_separator": right_separator,

                "inserted_span": placement["span"],
                "position": placement["position"]
            })

            cursor = offset

        pieces.append(
            original[cursor:]
        )

        modified = "".join(pieces)

        # Record original and final token observations.
        if own_records and tokenizer is not None:

            original_count = len(
                token_map(
                    original,
                    tokenizer
                )[0]
            )

            for record in own_records:
                record["original_token_count"] = (
                    original_count
                )

            final_token_observation(
                modified,
                own_records,
                tokenizer
            )

        output_documents.append({
            "pile_source_id": host["source_id"],
            "output_document_index": index,

            "text": modified,

            "original_sha256": sha(original),
            "modified_sha256": sha(modified),

            "insertion_count": len(own_records)
        })

        insertion_records.extend(
            own_records
        )

    return (
        output_documents,
        target_information,
        insertion_records
    )


# ============================================================
# 15. RUN EXPERIMENT AND SAVE JSONL
# ============================================================

def run(
    config,
    tokenizer=None
):
    """
    Run controlled insertion.

    Writes:
        modified_dataset.jsonl
        original_hosts.jsonl
        insertion_metadata.json

    JSONL contains exactly one document per line.
    """

    folder = Path(
        config["output_dir"]
    )

    if folder.exists():
        raise FileExistsError(
            f"Output directory already exists: {folder}"
        )

    hosts = load_hosts(
        config["pile"]
    )

    documents, targets, records = insert_targets(
        config,
        hosts,
        tokenizer=tokenizer
    )

    folder.mkdir(
        parents=True,
        exist_ok=False
    )

    # --------------------------------------------------------
    # A. MODIFIED DATASET — JSONL
    # --------------------------------------------------------

    dataset_path = (
        folder / "modified_dataset.jsonl"
    )

    with dataset_path.open(
        "x",
        encoding="utf-8"
    ) as file:

        for document in documents:

            file.write(
                json.dumps(
                    document,
                    ensure_ascii=False
                ) + "\n"
            )

    # --------------------------------------------------------
    # B. ORIGINAL DOCUMENTS — JSONL
    # --------------------------------------------------------

    originals_path = (
        folder / "original_hosts.jsonl"
    )

    with originals_path.open(
        "x",
        encoding="utf-8"
    ) as file:

        for host in hosts:

            file.write(
                json.dumps(
                    host,
                    ensure_ascii=False
                ) + "\n"
            )

    # --------------------------------------------------------
    # C. INSERTION METADATA — JSON
    # --------------------------------------------------------

    metadata_path = (
        folder / "insertion_metadata.json"
    )

    metadata = {
        "configuration": config,

        "dataset_format": "jsonl",

        "dataset_file": dataset_path.name,
        "original_hosts_file": originals_path.name,

        "total_documents": len(documents),
        "total_insertions": len(records),

        "targets": targets,
        "insertions": records
    }

    with metadata_path.open(
        "x",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # D. PRINT EXPERIMENT SUMMARY
    # --------------------------------------------------------

    modified_count = sum(
        document["insertion_count"] > 0
        for document in documents
    )

    print("\nCONTROLLED INSERTION COMPLETE")

    print(
        "Total Pile documents:",
        len(documents)
    )

    print(
        "Modified documents:",
        modified_count
    )

    print(
        "Total targets:",
        len(targets)
    )

    print(
        "Total insertions:",
        len(records)
    )

    print(
        "Modified dataset:",
        dataset_path
    )

    print(
        "Original documents:",
        originals_path
    )

    print(
        "Insertion metadata:",
        metadata_path
    )


# ============================================================
# 16. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    from transformers import AutoTokenizer

    # Load the tokenizer only once.
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["tokenizer_name"],
        revision=CONFIG.get(
            "tokenizer_revision"
        ),
        use_fast=True
    )

    # The insertion engine receives the tokenizer object.
    run(
        CONFIG,
        tokenizer=tokenizer
    )
