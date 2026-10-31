
#!/usr/bin/env python3

"""
CONTROLLED SPAN INSERTION INTO THE PILE DATASET
================================================

This script modifies the dataset only. It does not train models.

WORKFLOW
--------
1. Load selected documents from the original Pile dataset.
2. Obtain one or more target text spans.
3. Select existing Pile documents to receive each target.
4. Determine the insertion position within each document.
5. Insert target text without deleting original document text.
6. Validate insertion positions, frequency, and text preservation.
7. Save the modified dataset and detailed insertion metadata.

IMPORTANT DEFINITIONS
---------------------
Source ID:
    Zero-based row index in the original Pile dataset stream.

Output document index:
    Zero-based position within the loaded experimental subset.

Frequency:
    Number of distinct host documents receiving a target.
    Each target is inserted at most once per selected host.

Character offset:
    Insertion position in the ORIGINAL document.

Token offset:
    Character position derived from token boundaries in the
    ORIGINAL document. Retokenization may change boundaries.

Dataset position:
    Position within the loaded, unshuffled document sequence.
    Actual training exposure depends on the downstream trainer.

INSTALL
-------
pip install -U datasets transformers

OUTPUT
------
modified_dataset.json:
    Complete selected dataset, including unmodified documents.

insertion_metadata.json:
    Target descriptions, insertion positions, frequencies,
    and experimental configuration.
"""

import hashlib
import json
import random

from pathlib import Path
from itertools import islice
from collections import defaultdict


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================
#
# Researchers should normally modify only this section.
#
# The configuration separates:
#   1. Base dataset selection
#   2. Randomness and tokenizer settings
#   3. Output directory
#   4. Target spans
#   5. Insertion frequency and document selection
#   6. Within-document insertion positions
# ============================================================

CONFIG = {

    # --------------------------------------------------------
    # A. BASE DATASET
    # --------------------------------------------------------
    #
    # We insert target text into existing Pile documents.
    # No additional Pile documents are created.
    # --------------------------------------------------------

    "dataset": "EleutherAI/the_pile_deduplicated",

    # Optional named Hugging Face dataset configuration.
    "dataset_config": None,

    # Optional fixed revision for reproducibility.
    "dataset_revision": None,

    "split": "train",


    # --------------------------------------------------------
    # B. SELECT ORIGINAL PILE DOCUMENTS TO LOAD
    # --------------------------------------------------------
    #
    # This determines the experimental document pool.
    #
    # OPTION 1: Load first N documents
    #
    # {"mode": "first_n", "n": 100}
    #
    # OPTION 2: Load specific original source IDs
    #
    # {"mode": "source_ids", "ids": [0, 2, 5, 10]}
    #
    # OPTION 3: Load a source ID range
    #
    # {"mode": "source_range", "start": 100, "stop": 300}
    #
    # The stop index is exclusive.
    #
    # Loading documents does not necessarily mean
    # inserting target text into all of them.
    # --------------------------------------------------------

    "take": {
        "mode": "first_n",
        "n": 100
    },


    # --------------------------------------------------------
    # C. RANDOM SEED
    # --------------------------------------------------------
    #
    # Controls random host selection and random placement.
    # Does not shuffle the dataset.
    # --------------------------------------------------------

    "seed": 42,


    # --------------------------------------------------------
    # D. TOKENIZER
    # --------------------------------------------------------
    #
    # Used only when token-based placement is requested.
    # Must support fast-tokenizer offset mappings.
    # --------------------------------------------------------

    "tokenizer": "gpt2",
    "tokenizer_revision": None,


    # --------------------------------------------------------
    # E. OUTPUT DIRECTORY
    # --------------------------------------------------------
    #
    # Existing experiment directories are not overwritten.
    # Change this for every new experiment.
    # --------------------------------------------------------

    "output_dir": "pile_final_run_1",


    # --------------------------------------------------------
    # F. TARGET SPANS
    # --------------------------------------------------------
    #
    # Each target is configured independently.
    #
    # Different targets may have different:
    #   - Text sources
    #   - Frequencies
    #   - Host documents
    #   - Insertion positions
    #
    # Multiple different targets may share a host document.
    # --------------------------------------------------------

    "targets": [

        {
            # Unique target identifier.
            "id": "target_1",


            # ------------------------------------------------
            # F1. TARGET TEXT SOURCE
            # ------------------------------------------------
            #
            # OPTION A: Literal text
            #
            # "source": {
            #     "mode": "literal",
            #     "text": "my name is apratim"
            # }
            #
            # OPTION B: Text from another Hugging Face dataset
            #
            # "source": {
            #     "mode": "hf",
            #     "dataset": "SetFit/bbc-news",
            #     "dataset_config": None,
            #     "revision": None,
            #     "split": "train",
            #     "index": 0,
            #     "field": "text",
            #     "slice_start": 0,
            #     "slice_end": 150
            # }
            #
            # For HF mode:
            #   index       = external dataset row
            #   field       = field containing text
            #   slice_start = first selected character
            #   slice_end   = exclusive ending character
            #
            # Only the extracted span is inserted.
            # The external document is not added separately.
            # ------------------------------------------------

            "source": {
                "mode": "literal",
                "text": "my name is apratim"
            },


            # ------------------------------------------------
            # F2. FREQUENCY AND HOST-DOCUMENT SELECTION
            # ------------------------------------------------
            #
            # OPTION A: Explicit source IDs
            #
            # {"mode": "explicit", "source_ids": [2, 5, 10]}
            #
            # Frequency = 3 distinct documents.
            #
            # OPTION B: Random hosts
            #
            # {"mode": "random",
            #  "frequency": 10,
            #  "stage": "all"}
            #
            # Available stages:
            #   all
            #   early
            #   middle
            #   late
            #
            # OPTION C: Random hosts in a custom region
            #
            # {"mode": "random",
            #  "frequency": 10,
            #  "stage": "custom",
            #  "start_fraction": 0.20,
            #  "end_fraction": 0.80}
            #
            # OPTION D: Every Nth document
            #
            # {"mode": "every_n",
            #  "frequency": 10,
            #  "every_n": 5,
            #  "start_output_index": 0}
            #
            # OPTION E: Random hosts from a fixed range
            #
            # {"mode": "range",
            #  "frequency": 10,
            #  "start_output_index": 20,
            #  "stop_output_index": 80}
            #
            # Explicit selection uses ORIGINAL source IDs.
            # Range/every_n use LOADED output indices.
            # ------------------------------------------------

            "selection": {
                "mode": "explicit",
                "source_ids": [2, 5, 10]
            },


            # ------------------------------------------------
            # F3. DEFAULT WITHIN-DOCUMENT POSITION
            # ------------------------------------------------
            #
            # {"mode": "beginning"}
            # {"mode": "middle"}
            # {"mode": "end"}
            #
            # {"mode": "percentage", "value": 0.37}
            #
            # {"mode": "character_offset", "value": 50}
            #
            # {"mode": "random_character"}
            #
            # {"mode": "token_offset", "value": 25}
            #
            # {"mode": "random_token"}
            #
            # {"mode": "token_percentage", "value": 0.50}
            #
            # Percentages must be between 0 and 1.
            #
            # Every position is resolved against the ORIGINAL
            # host text, not text modified by other insertions.
            # ------------------------------------------------

            "position": {
                "mode": "percentage",
                "value": 0.37
            },


            # ------------------------------------------------
            # F4. DOCUMENT-SPECIFIC POSITION OVERRIDES
            # ------------------------------------------------
            #
            # Keys are ORIGINAL Pile source IDs as strings.
            #
            # These override the default position above.
            #
            # Example:
            #   Document 2  -> 10% of character length
            #   Document 5  -> character offset 30
            #   Document 10 -> end of document
            #
            # This enables positional diversity while keeping
            # the same target text and insertion frequency.
            # ------------------------------------------------

            "per_document_positions": {

                "2": {
                    "mode": "percentage",
                    "value": 0.10
                },

                "5": {
                    "mode": "character_offset",
                    "value": 30
                },

                "10": {
                    "mode": "end"
                }
            }
        }

        # Add more target dictionaries here, separated by commas.
    ]
}


# ============================================================
# 2. VALIDATION HELPERS
# ============================================================

def integer(value, name, minimum=0):
    """
    Require an integer greater than or equal to minimum.

    Boolean values are rejected even though bool is a
    subclass of int in Python.
    """

    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}, "
            f"got {value!r}"
        )

    return value


def fraction(value, name):
    """
    Validate a numeric fraction in the inclusive range [0, 1].
    """

    if (
        type(value) not in (int, float)
        or not 0 <= value <= 1
    ):
        raise ValueError(
            f"{name} must be numeric between 0 and 1"
        )

    return float(value)


def sha(text):
    """
    Return the SHA-256 checksum of UTF-8 encoded text.

    Checksums help identify the exact original and modified
    document contents used in an experiment.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# 3. STREAM HUGGING FACE DATASETS
# ============================================================

def hf_iterator(
    dataset,
    split="train",
    config=None,
    revision=None
):
    """
    Create a streaming iterator for a Hugging Face dataset.

    Streaming avoids loading the entire Pile into memory.

    Rows are read sequentially without explicit shuffling.
    """

    from datasets import load_dataset

    kwargs = {
        "path": dataset,
        "split": split,
        "streaming": True
    }

    if config is not None:
        kwargs["name"] = config

    if revision is not None:
        kwargs["revision"] = revision

    return iter(load_dataset(**kwargs))


# ============================================================
# 4. LOAD SELECTED ORIGINAL PILE DOCUMENTS
# ============================================================

def load_hosts(cfg):
    """
    Load the original Pile documents specified by cfg["take"].

    Returns a list of:
        {
            "source_id": original Pile row index,
            "original_text": original document content
        }

    Source ordering is preserved.
    """

    take = cfg["take"]
    mode = take["mode"]

    # Determine which original source IDs must be retained.
    if mode == "first_n":

        n = integer(take["n"], "take.n", 1)
        requested = set(range(n))

    elif mode == "source_ids":

        values = take["ids"]

        requested = {
            integer(i, "take.ids entry")
            for i in values
        }

        if not values or len(requested) != len(values):
            raise ValueError(
                "take.ids must contain distinct source IDs"
            )

    elif mode == "source_range":

        start = integer(take["start"], "take.start")

        stop = integer(
            take["stop"],
            "take.stop",
            start + 1
        )

        requested = set(range(start, stop))

    else:
        raise ValueError(
            f"Unknown take mode: {mode}"
        )

    rows = hf_iterator(
        cfg["dataset"],
        cfg.get("split", "train"),
        cfg.get("dataset_config"),
        cfg.get("dataset_revision")
    )

    hosts = []

    # Only stream through the largest requested source ID.
    #
    # Example:
    # Requested IDs = {2, 5, 10}
    #
    # Read rows 0 through 10, retaining only 2, 5, and 10.
    for idx, row in enumerate(
        islice(rows, max(requested) + 1)
    ):

        if idx not in requested:
            continue

        text = row.get("text")

        if not isinstance(text, str):
            raise ValueError(
                f"Pile row {idx} has no string text field"
            )

        hosts.append({
            "source_id": idx,
            "original_text": text
        })

    # Ensure the requested source rows actually existed.
    if len(hosts) != len(requested):
        raise ValueError(
            f"Requested {len(requested)} host rows, "
            f"but found {len(hosts)}"
        )

    return hosts


# ============================================================
# 5. LOAD TARGET TEXT
# ============================================================

def load_span(source):
    """
    Resolve a target span from literal text or an external
    Hugging Face dataset.

    For an external dataset, only the selected text span
    is extracted and inserted into existing Pile hosts.
    """

    mode = source["mode"]

    if mode == "literal":

        span = source["text"]

    elif mode == "hf":

        idx = integer(
            source["index"],
            "source.index"
        )

        rows = hf_iterator(
            source["dataset"],
            source.get("split", "train"),
            source.get("dataset_config"),
            source.get("revision")
        )

        # Skip to the requested external dataset row.
        row = next(
            islice(rows, idx, idx + 1),
            None
        )

        if row is None:
            raise ValueError(
                f"No row {idx} found in target dataset"
            )

        span = row.get(source["field"])

        if not isinstance(span, str):
            raise ValueError(
                "Chosen external source field "
                "does not contain text"
            )

        # Optional character-level extraction.
        start = source.get("slice_start")
        end = source.get("slice_end")

        if start is not None:
            integer(start, "slice_start")

        if end is not None:
            integer(end, "slice_end")

        span = span[start:end]

    else:
        raise ValueError(
            f"Unknown span source mode: {mode}"
        )

    if not isinstance(span, str) or not span:
        raise ValueError(
            "Target span must be nonempty text"
        )

    return span


# ============================================================
# 6. DATASET REGION SELECTION
# ============================================================

def stage_positions(selection, total):
    """
    Return eligible output indices for a dataset region.

    early/middle/late divide the LOADED document sequence
    into three consecutive regions.

    They do not represent actual training update steps.
    """

    stage = selection.get("stage", "all")

    bounds = {

        "all": (
            0,
            total
        ),

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

        a = fraction(
            selection["start_fraction"],
            "start_fraction"
        )

        b = fraction(
            selection["end_fraction"],
            "end_fraction"
        )

        if a >= b:
            raise ValueError(
                "start_fraction must be less than end_fraction"
            )

        return list(
            range(
                int(a * total),
                int(b * total)
            )
        )

    if stage not in bounds:
        raise ValueError(
            f"Unknown stage: {stage}"
        )

    return list(range(*bounds[stage]))


# ============================================================
# 7. CHOOSE HOST DOCUMENTS FOR A TARGET
# ============================================================

def choose_hosts(selection, hosts, span, rng):
    """
    Select distinct host documents receiving a target.

    Frequency is the number of DISTINCT selected hosts.

    Explicit mode uses original Pile source IDs.
    Range/every_n modes use loaded output indices.

    Hosts already containing the exact target substring
    are rejected or excluded to prevent preexisting target
    occurrences from distorting frequency measurements.
    """

    mode = selection["mode"]
    total = len(hosts)

    # Original source ID -> loaded output index.
    lookup = {
        host["source_id"]: i
        for i, host in enumerate(hosts)
    }

    # --------------------------------------------------------
    # EXPLICIT SOURCE IDS
    # --------------------------------------------------------

    if mode == "explicit":

        ids = selection["source_ids"]

        for x in ids:
            integer(x, "selection source ID")

        if len(set(ids)) != len(ids):
            raise ValueError(
                "Duplicate explicit source IDs"
            )

        if any(x not in lookup for x in ids):
            raise ValueError(
                "Explicit selection includes unloaded "
                "Pile source IDs"
            )

        chosen = [lookup[x] for x in ids]

    else:

        frequency = integer(
            selection["frequency"],
            "frequency"
        )

        # ----------------------------------------------------
        # RANDOM SELECTION FROM A STAGE
        # ----------------------------------------------------

        if mode == "random":

            candidates = stage_positions(
                selection,
                total
            )

        # ----------------------------------------------------
        # RANDOM SELECTION FROM A FIXED INDEX RANGE
        # ----------------------------------------------------

        elif mode == "range":

            a = integer(
                selection["start_output_index"],
                "range start"
            )

            b = integer(
                selection["stop_output_index"],
                "range stop",
                a + 1
            )

            if b > total:
                raise ValueError(
                    "Range exceeds number of loaded documents"
                )

            candidates = list(range(a, b))

        # ----------------------------------------------------
        # EVERY NTH DOCUMENT
        # ----------------------------------------------------

        elif mode == "every_n":

            a = integer(
                selection.get("start_output_index", 0),
                "every_n start"
            )

            every = integer(
                selection["every_n"],
                "every_n",
                1
            )

            candidates = list(
                range(a, total, every)
            )

        else:
            raise ValueError(
                f"Unknown selection mode: {mode}"
            )

        # Remove candidate hosts already containing the target.
        #
        # For example, if "my name is apratim" already
        # occurs in a host, it is not eligible for another
        # controlled occurrence of that target.
        candidates = [
            i for i in candidates
            if span not in hosts[i]["original_text"]
        ]

        if frequency > len(candidates):
            raise ValueError(
                f"Frequency {frequency} exceeds "
                f"{len(candidates)} eligible hosts"
            )

        if mode in ("random", "range"):

            # Sampling without replacement ensures
            # distinct host documents.
            chosen = rng.sample(
                candidates,
                frequency
            )

        else:

            # Every-N selection retains deterministic order.
            chosen = candidates[:frequency]

    # Explicit selections must also reject preexisting spans.
    for i in chosen:

        if span in hosts[i]["original_text"]:
            raise ValueError(
                "Target already appears in original "
                f"host {hosts[i]['source_id']}"
            )

    return sorted(chosen)


# ============================================================
# 8. TOKENIZER-BASED POSITION SUPPORT
# ============================================================

class TokenOffsets:
    """
    Convert original token boundaries into character offsets.

    A fast tokenizer provides offset_mapping, describing
    where each token lies in the original character string.

    An N-token document has N+1 conceptual boundaries:
        0 -> before first token
        N -> after final token

    Interior overlapping offsets are rejected because
    they cannot be represented as unambiguous boundaries.

    Tokenization of the modified document can differ from
    tokenization of the original document.
    """

    def __init__(self, cfg):

        self.cfg = cfg
        self.tokenizer = None

    def boundaries(self, text):
        """
        Return character offsets for token boundaries.

        None indicates an ambiguous overlapping boundary.
        """

        # Lazily initialize the tokenizer.
        if self.tokenizer is None:

            from transformers import AutoTokenizer

            kwargs = {
                "use_fast": True
            }

            if self.cfg.get("tokenizer_revision"):
                kwargs["revision"] = (
                    self.cfg["tokenizer_revision"]
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg["tokenizer"],
                **kwargs
            )

            if not self.tokenizer.is_fast:
                raise ValueError(
                    "Tokenizer must be fast and support "
                    "offset_mapping"
                )

        # Tokenize without adding special tokens or truncating.
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False
        )

        offsets = encoded["offset_mapping"]

        # Beginning of original document.
        result = [0]

        for k in range(1, len(offsets)):

            prev_end = offsets[k - 1][1]
            next_start = offsets[k][0]

            if prev_end > next_start:

                # Overlapping offsets are ambiguous.
                result.append(None)

            else:

                # Boundary derived from preceding token end.
                result.append(prev_end)

        # End of original document.
        result.append(len(text))

        return result


# ============================================================
# 9. RESOLVE A WITHIN-DOCUMENT INSERTION OFFSET
# ============================================================

def position_offset(
    text,
    specification,
    rng,
    token_helper
):
    """
    Convert a position specification into a character offset
    within the ORIGINAL document.

    Supported modes:
        beginning
        middle
        end
        percentage
        character_offset
        random_character
        token_offset
        random_token
        token_percentage

    All returned offsets are original-text character indices.
    """

    mode = specification["mode"]
    length = len(text)

    # --------------------------------------------------------
    # DOCUMENT BEGINNING
    # --------------------------------------------------------

    if mode == "beginning":
        return 0

    # --------------------------------------------------------
    # DOCUMENT MIDDLE (CHARACTER-BASED)
    # --------------------------------------------------------

    if mode == "middle":
        return length // 2

    # --------------------------------------------------------
    # DOCUMENT END
    # --------------------------------------------------------

    if mode == "end":
        return length

    # --------------------------------------------------------
    # RANDOM CHARACTER BOUNDARY
    # --------------------------------------------------------
    #
    # Includes both the beginning and end of the document.

    if mode == "random_character":
        return rng.randint(0, length)

    # --------------------------------------------------------
    # FRACTION OF ORIGINAL CHARACTER LENGTH
    # --------------------------------------------------------

    if mode == "percentage":

        value = fraction(
            specification["value"],
            "percentage"
        )

        # This can produce an offset inside a word.
        return round(length * value)

    # --------------------------------------------------------
    # EXACT CHARACTER OFFSET
    # --------------------------------------------------------

    if mode == "character_offset":

        offset = integer(
            specification["value"],
            "character_offset"
        )

        if offset > length:
            raise ValueError(
                f"Character offset {offset} exceeds "
                f"document length {length}"
            )

        return offset

    # --------------------------------------------------------
    # TOKEN-BASED POSITIONS
    # --------------------------------------------------------

    if mode in (
        "token_offset",
        "random_token",
        "token_percentage"
    ):

        boundaries = token_helper.boundaries(text)

        token_count = len(boundaries) - 1

        if mode == "token_offset":

            k = integer(
                specification["value"],
                "token_offset"
            )

        elif mode == "token_percentage":

            value = fraction(
                specification["value"],
                "token_percentage"
            )

            k = round(token_count * value)

        else:

            # Only choose valid, non-overlapping boundaries.
            valid = [
                k
                for k, value in enumerate(boundaries)
                if value is not None
            ]

            k = rng.choice(valid)

        if k > token_count:
            raise ValueError(
                f"Token offset {k} exceeds original "
                f"token count {token_count}"
            )

        if boundaries[k] is None:
            raise ValueError(
                f"Token boundary {k} overlaps in tokenizer "
                "offset mapping; choose another boundary"
            )

        return boundaries[k]

    raise ValueError(
        f"Unknown position mode: {mode}"
    )


# ============================================================
# 10. MAIN INSERTION ENGINE
# ============================================================

def apply_insertions(
    cfg,
    hosts,
    resolved_spans=None
):
    """
    Apply all configured target insertions.

    PHASE A: PLAN
        Resolve targets, choose hosts, and compute positions
        using the ORIGINAL host documents.

    PHASE B: APPLY
        Apply all planned insertions, preserving original text
        and document ordering.

    PHASE C: VALIDATE
        Verify recorded span positions, original-text
        reconstruction, and per-target insertion frequency.

    Multiple targets may share the same host document.

    Returns
    -------
    output_documents:
        Modified and unmodified document records.

    target_info:
        Target-level experimental metadata.

    insertion_records:
        Individual insertion records.
    """

    rng = random.Random(cfg["seed"])

    if not hosts:
        raise ValueError("No hosts")

    # Verify distinct, ascending original source IDs.
    source_ids = [
        host["source_id"]
        for host in hosts
    ]

    if source_ids != sorted(set(source_ids)):
        raise ValueError(
            "Hosts must have distinct source IDs "
            "in original Pile order"
        )

    # Preserve original texts throughout planning.
    originals = [
        host["original_text"]
        for host in hosts
    ]

    token_helper = TokenOffsets(cfg)

    # Maps output document index -> planned insertions.
    planned = defaultdict(list)

    target_info = []
    seen_ids = set()

    # ========================================================
    # PHASE A: PLAN ALL TARGET INSERTIONS
    # ========================================================

    for target_index, target in enumerate(cfg["targets"]):

        target_id = target["id"]

        if target_id in seen_ids:
            raise ValueError(
                f"Duplicate target ID: {target_id}"
            )

        seen_ids.add(target_id)

        # ----------------------------------------------------
        # A1. RESOLVE TARGET TEXT
        # ----------------------------------------------------

        if (
            resolved_spans is not None
            and target_id in resolved_spans
        ):
            span = resolved_spans[target_id]

        else:
            span = load_span(target["source"])

        if not isinstance(span, str) or not span:
            raise ValueError(
                "Resolved span must be nonempty"
            )

        # ----------------------------------------------------
        # A2. SELECT DISTINCT HOST DOCUMENTS
        # ----------------------------------------------------

        chosen = choose_hosts(
            target["selection"],
            hosts,
            span,
            rng
        )

        # ----------------------------------------------------
        # A3. RESOLVE EACH HOST'S INSERTION POSITION
        # ----------------------------------------------------

        for i in chosen:

            host = hosts[i]

            # Document-specific overrides take precedence
            # over the target's default position.
            position_spec = target.get(
                "per_document_positions",
                {}
            ).get(
                str(host["source_id"]),
                target["position"]
            )

            # Compute offset using original document text.
            offset = position_offset(
                originals[i],
                position_spec,
                rng,
                token_helper
            )

            # Store the insertion rather than applying it
            # immediately. This prevents earlier insertions
            # from shifting later original-text offsets.
            planned[i].append({

                "target_id": target_id,

                "target_order": target_index,

                "span": span,

                "offset": offset,

                "position": position_spec
            })

        # ----------------------------------------------------
        # A4. RECORD TARGET-LEVEL METADATA
        # ----------------------------------------------------

        target_info.append({

            "id": target_id,

            "source": target["source"],

            "span_sha256": sha(span),

            "span": span,

            "frequency": len(chosen),

            "host_source_ids": [
                hosts[i]["source_id"]
                for i in chosen
            ]
        })

    # ========================================================
    # PHASE B: APPLY INSERTIONS
    # ========================================================

    output_documents = []
    insertion_records = []

    # Process every host in original dataset order.
    for i, host in enumerate(hosts):

        original_text = originals[i]

        # Sort insertions by original character position.
        #
        # If two targets share the same offset, their
        # configuration order breaks the tie.
        placements = sorted(
            planned[i],
            key=lambda p: (
                p["offset"],
                p["target_order"]
            )
        )

        # Build modified text from fragments to avoid
        # repeatedly copying the entire document.
        pieces = []

        # Position already consumed in ORIGINAL text.
        cursor = 0

        # Number of characters already emitted in output.
        output_length = 0

        # Records for insertions into this particular host.
        own_records = []

        for placement in placements:

            offset = placement["offset"]

            # ------------------------------------------------
            # B1. COPY ORIGINAL TEXT BEFORE INSERTION
            # ------------------------------------------------

            segment = original_text[cursor:offset]

            pieces.append(segment)
            output_length += len(segment)

            # ------------------------------------------------
            # B2. ADD INSERTION SEPARATORS
            # ------------------------------------------------
            #
            # Separate target text from the original context.
            #
            # No left separator at document beginning.
            # No right separator at original document end.
            #
            # Separators are additional text and are recorded
            # separately for exact reconstruction.
            left_separator = (
                "\n\n"
                if offset > 0
                else ""
            )

            right_separator = (
                "\n\n"
                if offset < len(original_text)
                else ""
            )

            # ------------------------------------------------
            # B3. CALCULATE TARGET OFFSETS IN MODIFIED TEXT
            # ------------------------------------------------
            #
            # start is inclusive; end is exclusive.
            span_start = (
                output_length
                + len(left_separator)
            )

            span_end = (
                span_start
                + len(placement["span"])
            )

            # ------------------------------------------------
            # B4. APPEND INSERTED CONTENT
            # ------------------------------------------------

            pieces.extend((
                left_separator,
                placement["span"],
                right_separator
            ))

            output_length += (
                len(left_separator)
                + len(placement["span"])
                + len(right_separator)
            )

            # ------------------------------------------------
            # B5. SAVE INSERTION DETAILS
            # ------------------------------------------------

            record = {

                "target_id": placement["target_id"],

                "pile_source_id": host["source_id"],

                "output_document_index": i,

                "position_spec": placement["position"],

                "original_character_offset": offset,

                "inserted_start_char": span_start,

                "inserted_end_char": span_end,

                "left_separator": left_separator,

                "right_separator": right_separator,

                "inserted_span": placement["span"]
            }

            own_records.append(record)

            # Advance through original text only.
            cursor = offset

        # ----------------------------------------------------
        # B6. APPEND REMAINING ORIGINAL TEXT
        # ----------------------------------------------------

        pieces.append(original_text[cursor:])

        modified_text = "".join(pieces)

        # ====================================================
        # PHASE C: VERIFY ORIGINAL-TEXT PRESERVATION
        # ====================================================
        #
        # The original document must be recoverable by
        # removing every inserted span and its separators.
        #
        # Reverse-order removal is necessary because deleting
        # later insertions does not shift earlier offsets.
        # ====================================================

        reconstructed = modified_text

        for record in reversed(own_records):

            start = record["inserted_start_char"]
            end = record["inserted_end_char"]

            # Verify exact target text at its recorded offset.
            if (
                reconstructed[start:end]
                != record["inserted_span"]
            ):
                raise RuntimeError(
                    "Inserted span or offset mismatch"
                )

            # Include the recorded separators in removal.
            remove_start = (
                start
                - len(record["left_separator"])
            )

            remove_end = (
                end
                + len(record["right_separator"])
            )

            reconstructed = (
                reconstructed[:remove_start]
                + reconstructed[remove_end:]
            )

        # Exact character-for-character comparison.
        if reconstructed != original_text:
            raise RuntimeError(
                "Original text was not preserved in "
                f"Pile document {host['source_id']}"
            )

        # Store the complete document, even if unmodified.
        output_documents.append({

            "pile_source_id": host["source_id"],

            "output_document_index": i,

            "text": modified_text,

            "original_sha256": sha(original_text),

            "modified_sha256": sha(modified_text),

            "insertion_count": len(own_records)
        })

        insertion_records.extend(own_records)

    # ========================================================
    # PHASE D: VERIFY PER-TARGET FREQUENCY
    # ========================================================
    #
    # Check that:
    #   1. Recorded insertion count matches planned frequency.
    #   2. Every insertion for a target uses a distinct host.
    # ========================================================

    for target in target_info:

        actual = [
            record
            for record in insertion_records
            if record["target_id"] == target["id"]
        ]

        unique_hosts = {
            record["pile_source_id"]
            for record in actual
        }

        if (
            len(actual) != target["frequency"]
            or len(unique_hosts) != len(actual)
        ):
            raise RuntimeError(
                "Frequency or distinct-host "
                "validation failed"
            )

    return (
        output_documents,
        target_info,
        insertion_records
    )


# ============================================================
# 11. RUN EXPERIMENT AND SAVE OUTPUTS
# ============================================================

def main():
    """
    Execute the controlled insertion experiment.

    Outputs:
        modified_dataset.json
        insertion_metadata.json

    The script does not modify the original Hugging Face
    dataset or perform any model training.
    """

    output_dir = Path(CONFIG["output_dir"])

    # --------------------------------------------------------
    # PREVENT ACCIDENTAL EXPERIMENT OVERWRITING
    # --------------------------------------------------------

    if output_dir.exists():
        raise FileExistsError(
            "Choose a new output_dir. "
            f"Already exists: {output_dir}"
        )

    # --------------------------------------------------------
    # LOAD ORIGINAL HOST DOCUMENTS
    # --------------------------------------------------------

    print("Loading selected Pile documents...")

    hosts = load_hosts(CONFIG)

    print(
        f"Loaded {len(hosts)} existing Pile documents."
    )

    # --------------------------------------------------------
    # PERFORM CONTROLLED INSERTIONS AND VALIDATION
    # --------------------------------------------------------

    print("Applying controlled insertions...")

    documents, targets, insertions = apply_insertions(
        CONFIG,
        hosts
    )

    # ========================================================
    # OUTPUT 1: COMPLETE MODIFIED DATASET
    # ========================================================
    #
    # The output contains every loaded host document,
    # including documents that received no insertions.
    #
    # Documents remain in original source order.
    #
    # The dataset is saved as ONE JSON object.
    # ========================================================

    result = {

        "dataset": CONFIG["dataset"],

        "configuration": CONFIG,

        "document_order":
            "original_pile_stream_order_no_shuffle",

        "training_time_exposure":
            "not_controlled_by_dataset_editing",

        "total_documents": len(documents),

        "total_insertions": len(insertions),

        "documents": documents
    }

    # ========================================================
    # OUTPUT 2: INSERTION METADATA
    # ========================================================
    #
    # Records experimental configuration, target frequencies,
    # original host IDs, and exact insertion positions.
    #
    # The metadata is intended to support reproducibility
    # and subsequent memorization analysis.
    # ========================================================

    metadata = {

        "configuration": CONFIG,

        "targets": targets,

        "insertions": insertions,

        "total_documents": len(documents),

        "total_insertions": len(insertions),

        "token_note": (
            "Token positions use original tokenizer offsets. "
            "Insertion may change tokenization at boundaries. "
            "Exact post-insertion token identity is not guaranteed."
        ),

        "training_note": (
            "The downstream trainer must preserve record order "
            "to reproduce intended exposure order."
        )
    }

    # --------------------------------------------------------
    # CREATE EXPERIMENT OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    dataset_file = (
        output_dir / "modified_dataset.json"
    )

    metadata_file = (
        output_dir / "insertion_metadata.json"
    )

    # --------------------------------------------------------
    # SAVE COMPLETE MODIFIED DATASET
    # --------------------------------------------------------
    #
    # ensure_ascii=False preserves Unicode characters.
    # indent=2 creates human-readable JSON.
    # "x" prevents accidental file overwriting.
    with dataset_file.open(
        "x",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # SAVE INSERTION METADATA
    # --------------------------------------------------------

    with metadata_file.open(
        "x",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # PRINT FINAL EXPERIMENT SUMMARY
    # ========================================================

    print("\nCONTROLLED INSERTION COMPLETE")

    print(
        "Total Pile documents:",
        len(documents)
    )

    print(
        "Total insertions:",
        len(insertions)
    )

    print(
        "Modified dataset:",
        dataset_file
    )

    print(
        "Insertion metadata:",
        metadata_file
    )

    print("\nINSERTION DETAILS")

    for record in insertions:

        print(
            f"Target: {record['target_id']} | "
            f"Pile document: {record['pile_source_id']} | "
            f"Original character offset: "
            f"{record['original_character_offset']}"
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================
#
# Execute main() only when running this script directly.
#
# Importing the module from another Python script will not
# automatically download datasets or create output files.
# ============================================================

if __name__ == "__main__":
    main()
