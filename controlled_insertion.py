
#!/usr/bin/env python3

"""
CONTROLLED SPAN INSERTION INTO THE PILE DATASET
================================================

PROJECT OBJECTIVE
-----------------

HOW THE FRAMEWORK WORKS
-----------------------
1. Load selected documents from the original Pile dataset.
2. Obtain the target text span(s).
3. Select which existing Pile documents receive each target.
4. Determine the insertion position inside each document.
5. Insert the target without deleting original document text.
6. Validate that the insertion was performed correctly.
7. Save the modified dataset and detailed insertion metadata.


EXAMPLE
-------
Suppose we load 100 Pile documents and want to insert:

    "my name is apratim"

into documents with original source IDs 2, 5, and 10.

We can independently specify:

    Document 2  -> insert at 10% of its original length
    Document 5  -> insert at character offset 30
    Document 10 -> insert at the end

All other documents remain unchanged.

The original order of Pile documents is preserved.


IMPORTANT DEFINITIONS
---------------------
Source ID:
    Zero-based row index in the original Hugging Face
    Pile dataset stream.

Output document index:
    Zero-based index within the subset loaded by this script.

Frequency:
    Number of DISTINCT host documents receiving one insertion
    of a particular target span.

Character offset:
    Position between characters of the ORIGINAL document.
    Offset 0 means before the first character.

Token offset:
    Insertion boundary derived from the tokenizer's offset
    mapping on the ORIGINAL document.

    The modified text may tokenize differently afterward.

Dataset position:
    Position in the loaded, unshuffled dataset sequence.
    It does not directly control optimizer update steps.


DEPENDENCIES
------------
pip install -U datasets transformers

RUN
---
python controlled_insertion_final.py

OUTPUTS
-------
1. modified_dataset.json
   Complete selected dataset, including modified and
   unmodified Pile documents.

2. insertion_metadata.json
   Experimental configuration, target spans, positions,
   frequencies, and insertion records.

import hashlib
import json
import random

from pathlib import Path
from itertools import islice
from collections import defaultdict


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

CONFIG = {

    # --------------------------------------------------------
    # A. BASE DATASET
    # --------------------------------------------------------
    #
    # The original dataset into which target spans are inserted.
    #
    # IMPORTANT:
    # We modify the TEXT of existing documents.
    # We do not create additional Pile documents.
    #
    # Consequently, the number of output documents is equal
    # to the number of loaded host documents.
    # --------------------------------------------------------

    "dataset": "EleutherAI/the_pile_deduplicated",

    # Optional Hugging Face dataset configuration.
    # Keep None when no named configuration is required.
    "dataset_config": None,

    # Optional dataset commit/revision.
    #
    # Pinning a revision improves experiment reproducibility
    # because the underlying dataset could change over time.
    "dataset_revision": None,

    "split": "train",


    # --------------------------------------------------------
    # B. SELECT WHICH PILE DOCUMENTS TO LOAD
    # --------------------------------------------------------
    #
    # This setting determines the experimental dataset subset.
    #
    # It is NOT the same as choosing which documents receive
    # target insertions.
    #
    # First we load the host-document pool.
    # Later, each target selects recipients from that pool.
    #
    # Supported modes:
    #
    # 1. First N documents:
    #
    #    {"mode": "first_n", "n": 100}
    #
    #    Loads source IDs 0 through 99.
    #
    # 2. Specific original source IDs:
    #
    #    {"mode": "source_ids", "ids": [0, 2, 5, 10]}
    #
    #    Loads only these documents.
    #    Output order remains 0, 2, 5, 10.
    #
    # 3. Source ID range:
    #
    #    {"mode": "source_range",
    #     "start": 100,
    #     "stop": 300}
    #
    #    Loads source IDs 100 through 299.
    #    The stop index is exclusive.
    #
    # Note:
    # Source IDs refer to original stream indices, not
    # the positions of documents after selection.
    # --------------------------------------------------------

    "take": {
        "mode": "first_n",
        "n": 100
    },


    # --------------------------------------------------------
    # C. RANDOM SEED
    # --------------------------------------------------------
    #
    # Ensures that random host selection and random insertion
    # positions are reproducible with the same inputs.
    #
    # The seed does NOT shuffle the dataset.
    #
    # A local random-number generator is created rather than
    # changing Python's global random state.
    # --------------------------------------------------------

    "seed": 42,


    # --------------------------------------------------------
    # D. TOKENIZER CONFIGURATION
    # --------------------------------------------------------
    #
    # Required only for token-based placement modes:
    #
    #   token_offset
    #   random_token
    #   token_percentage
    #
    # The tokenizer is loaded lazily, meaning it is not
    # downloaded or initialized unless token-based placement
    # is actually requested.
    #
    # A FAST tokenizer is required because we need character
    # offset mappings for original text tokens.
    # --------------------------------------------------------

    "tokenizer": "gpt2",
    "tokenizer_revision": None,


    # --------------------------------------------------------
    # E. EXPERIMENT OUTPUT DIRECTORY
    # --------------------------------------------------------
    #
    # Use a different directory for each experiment.
    #
    # Existing directories are intentionally rejected to
    # avoid accidentally overwriting previous results.
    # --------------------------------------------------------

    "output_dir": "pile_final_run_1",


    # --------------------------------------------------------
    # F. TARGET SPANS
    # --------------------------------------------------------
    #
    # Each dictionary describes ONE independently controlled
    # target text span.
    #
    # Multiple targets are supported.
    #
    # Example:
    #   target_1 -> appears in 3 documents
    #   target_2 -> appears in 10 documents
    #
    # Different targets may also appear in the same document.
    # --------------------------------------------------------

    "targets": [

        {
            # Unique identifier used in the metadata.
            "id": "target_1",


            # ------------------------------------------------
            # F1. TARGET SPAN SOURCE
            # ------------------------------------------------
            #
            # OPTION A: Directly specified text.
            #
            # This is useful for a controlled canary,
            # fictional statement, or other predefined span.
            #
            # OPTION B: Extract text from another HF dataset.
            #
            # Example:
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
            # Here:
            #   index = external dataset row
            #   field = column containing the text
            #   slice_start = first selected character
            #   slice_end = exclusive ending character
            #
            # Only the extracted span is inserted into Pile.
            # The external document itself is not added
            # as a separate training record.
            # ------------------------------------------------

            "source": {
                "mode": "literal",
                "text": "my name is apratim"
            },


            # ------------------------------------------------
            # F2. INSERTION FREQUENCY AND DATASET POSITION
            # ------------------------------------------------
            #
            # This configuration chooses WHICH Pile documents
            # receive the target text.
            #
            # Experimental frequency is the number of distinct
            # host documents receiving this target.
            #
            # Supported selection modes:
            #
            # OPTION A: Explicit source IDs
            #
            # {"mode": "explicit",
            #  "source_ids": [2, 5, 10]}
            #
            # Frequency is 3.
            #
            # OPTION B: Randomly selected hosts
            #
            # {"mode": "random",
            #  "frequency": 10,
            #  "stage": "all"}
            #
            # Stage can be:
            #   all    -> entire loaded dataset
            #   early  -> first third
            #   middle -> middle third
            #   late   -> final third
            #
            # OPTION C: Custom fractional dataset region
            #
            # {"mode": "random",
            #  "frequency": 10,
            #  "stage": "custom",
            #  "start_fraction": 0.20,
            #  "end_fraction": 0.80}
            #
            # Select 10 random documents from the region
            # spanning 20% to 80% of the loaded dataset.
            #
            # OPTION D: Every Nth eligible document
            #
            # {"mode": "every_n",
            #  "frequency": 10,
            #  "every_n": 5,
            #  "start_output_index": 0}
            #
            # Select from indices:
            #   0, 5, 10, 15, ...
            #
            # The first 10 eligible candidates are selected.
            #
            # OPTION E: Random selection within index range
            #
            # {"mode": "range",
            #  "frequency": 10,
            #  "start_output_index": 20,
            #  "stop_output_index": 80}
            #
            # Select 10 distinct documents randomly from
            # output indices 20 through 79.
            #
            # IMPORTANT:
            # Explicit selection uses ORIGINAL source IDs.
            # Range/every_n use LOADED output indices.
            # ------------------------------------------------

            "selection": {
                "mode": "explicit",
                "source_ids": [2, 5, 10]
            },


            # ------------------------------------------------
            # F3. DEFAULT POSITION INSIDE EACH DOCUMENT
            # ------------------------------------------------
            #
            # Controls WHERE the target appears in a selected
            # host document.
            #
            # All offsets are calculated using original text.
            #
            # Supported modes:
            #
            # {"mode": "beginning"}
            # {"mode": "middle"}
            # {"mode": "end"}
            #
            # {"mode": "percentage", "value": 0.37}
            #   Insert around 37% of the character length.
            #
            # {"mode": "character_offset", "value": 50}
            #   Insert after the first 50 characters.
            #
            # {"mode": "random_character"}
            #   Random character boundary, including ends.
            #
            # {"mode": "token_offset", "value": 25}
            #   Insert at tokenizer-derived boundary 25.
            #
            # {"mode": "random_token"}
            #   Random valid token boundary.
            #
            # {"mode": "token_percentage", "value": 0.50}
            #   Insert around 50% of original token count.
            #
            # Percentages range from 0.0 through 1.0.
            #
            # This default is applied unless a per-document
            # override is specified below.
            # ------------------------------------------------

            "position": {
                "mode": "percentage",
                "value": 0.37
            },


            # ------------------------------------------------
            # F4. DOCUMENT-SPECIFIC POSITION OVERRIDES
            # ------------------------------------------------
            #
            # Different selected documents can receive
            # the same target at different positions.
            #
            # Dictionary keys are original Pile source IDs
            # written as strings.
            #
            # These settings OVERRIDE the default position.
            #
            # In this example:
            #
            # Document 2:
            #   Insert at 10% of original character length.
            #
            # Document 5:
            #   Insert at character offset 30.
            #
            # Document 10:
            #   Insert at the end.
            #
            # The feature is useful when studying effects
            # of positional diversity on memorization.
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

        # Add additional target dictionaries here.
        #
        # Each target needs a unique ID.
        # Different targets can use different frequencies,
        # host selections, and insertion positions.
    ]
}


# ============================================================
# 2. BASIC VALIDATION UTILITIES
# ============================================================
#
# These helpers validate configuration values before they
# affect document selection or insertion.
#
# Early validation helps prevent invalid experiments
# from silently producing incorrect datasets.
# ============================================================

def integer(value, name, minimum=0):
    """
    Validate an integer-valued configuration parameter.

    Parameters
    ----------
    value:
        Input value to validate.

    name:
        Parameter name shown in error messages.

    minimum:
        Smallest permitted integer value.

    Returns
    -------
    int:
        The validated integer.

    Notes
    -----
    Python considers bool a subclass of int.

    We deliberately reject True and False because boolean
    values should not be accepted as document counts,
    source IDs, or character offsets.
    """

    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}, "
            f"got {value!r}"
        )

    return value


def fraction(value, name):
    """
    Validate a fractional position between 0 and 1.

    Fractions are used for:
        - Within-document percentage positions
        - Token-percentage positions
        - Custom dataset-region boundaries

    For example:
        0.0 -> beginning
        0.5 -> midpoint
        1.0 -> end

    Returns the value as a float.
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
    Compute the SHA-256 checksum of UTF-8 encoded text.

    Why store checksums?
    --------------------
    They provide stable fingerprints of original and
    modified document contents.

    These fingerprints can later help researchers verify
    which exact text was used in an experiment.

    This function does not encrypt the text.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# 3. STREAM A HUGGING FACE DATASET
# ============================================================

def hf_iterator(
    dataset,
    split="train",
    config=None,
    revision=None
):
    """
    Create a streaming iterator over a Hugging Face dataset.

    Why streaming?
    --------------
    The Pile is very large.

    Loading the complete dataset into memory would be
    unnecessary when an experiment only needs a subset.

    Streaming allows us to access rows sequentially
    without downloading the complete dataset first.

    Parameters
    ----------
    dataset:
        Hugging Face dataset identifier.

    split:
        Dataset split to read.

    config:
        Optional named dataset configuration.

    revision:
        Optional dataset commit/revision for reproducibility.

    Returns
    -------
    iterator:
        Sequential iterator over dataset rows.

    Important
    ---------
    Streaming preserves source order unless the underlying
    dataset itself uses a different ordering.

    This function never explicitly shuffles records.
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
# 4. LOAD ORIGINAL PILE HOST DOCUMENTS
# ============================================================

def load_hosts(cfg):
    """
    Load the subset of original Pile documents requested
    by CONFIG["take"].

    A host document is an existing Pile document that may
    receive one or more target text insertions.

    This function only loads documents. It does not yet
    decide which ones will be modified.

    Each returned host contains:
        source_id
        original_text

    The source_id is preserved so results can be traced
    back to the original dataset stream.

    Returns
    -------
    list[dict]:
        Selected host documents in original source order.
    """

    take = cfg["take"]
    mode = take["mode"]

    # --------------------------------------------------------
    # Determine which original source IDs must be loaded.
    # --------------------------------------------------------

    if mode == "first_n":

        # Example: n=100 -> source IDs 0 through 99.
        n = integer(take["n"], "take.n", 1)

        requested = set(range(n))

    elif mode == "source_ids":

        # Load only the explicitly requested source IDs.
        values = take["ids"]

        requested = {
            integer(i, "take.ids entry")
            for i in values
        }

        # Duplicate IDs are rejected to avoid ambiguity.
        if not values or len(requested) != len(values):
            raise ValueError(
                "take.ids must contain distinct source IDs"
            )

    elif mode == "source_range":

        # Source range uses an exclusive stop index.
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

    # Create a streaming iterator over the base dataset.
    rows = hf_iterator(
        cfg["dataset"],
        cfg.get("split", "train"),
        cfg.get("dataset_config"),
        cfg.get("dataset_revision")
    )

    hosts = []

    # We only need to stream up to the largest requested ID.
    #
    # Example:
    # Requested source IDs = {2, 5, 10}
    #
    # We must read through index 10, but we only retain
    # rows 2, 5, and 10.
    #
    # Streaming still requires visiting preceding rows.
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

    # Ensure every requested document was available.
    #
    # Without this check, experiments could silently use
    # fewer documents than requested.
    if len(hosts) != len(requested):
        raise ValueError(
            f"Requested {len(requested)} host rows, "
            f"but found {len(hosts)}"
        )

    return hosts


# ============================================================
# 5. OBTAIN THE TARGET TEXT SPAN
# ============================================================

def load_span(source):
    """
    Resolve the text span to be inserted.

    Two modes are supported:

    1. literal:
       Use manually specified target text.

    2. hf:
       Extract text from a selected row and column of
       another Hugging Face dataset.

    The returned span is inserted INTO existing Pile
    documents rather than becoming a standalone document.

    This distinction matters because the experiment studies
    memorization under controlled contextual exposure.
    """

    mode = source["mode"]

    if mode == "literal":

        # Directly use the text written in the configuration.
        span = source["text"]

    elif mode == "hf":

        # Select the row from the external dataset.
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

        # islice skips to the requested row.
        #
        # next(..., None) avoids StopIteration if the
        # requested row does not exist.
        row = next(
            islice(rows, idx, idx + 1),
            None
        )

        if row is None:
            raise ValueError(
                f"No row {idx} found in target dataset"
            )

        # Read the selected text field from the external row.
        span = row.get(source["field"])

        if not isinstance(span, str):
            raise ValueError(
                "Chosen external source field "
                "does not contain text"
            )

        # Optional character slicing of the external text.
        #
        # For example:
        # slice_start=0, slice_end=150
        #
        # extracts the first 150 characters.
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

    # Empty targets are not meaningful insertion objects.
    if not isinstance(span, str) or not span:
        raise ValueError(
            "Target span must be nonempty text"
        )

    return span


# ============================================================
# 6. DETERMINE ELIGIBLE DATASET REGIONS
# ============================================================

def stage_positions(selection, total):
    """
    Return eligible output document indices for a dataset
    region.

    This function is used when random host selection
    specifies an insertion stage.

    Stages
    ------
    all:
        All loaded documents.

    early:
        First third of the loaded document sequence.

    middle:
        Middle third of the loaded document sequence.

    late:
        Final third of the loaded document sequence.

    custom:
        Arbitrary fractional region.

    Example
    -------
    Suppose total = 300.

    early  -> indices 0 through 99
    middle -> indices 100 through 199
    late   -> indices 200 through 299

    IMPORTANT
    ---------
    These regions refer to the LOADED dataset order.

    They do not automatically correspond to early or late
    training steps because the downstream training pipeline
    may shuffle documents or repeat epochs.
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

        # Convert fractional boundaries to output indices.
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

        # Integer conversion creates discrete index bounds.
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
# 7. SELECT HOST DOCUMENTS FOR A TARGET
# ============================================================

def choose_hosts(selection, hosts, span, rng):
    """
    Select the host documents that will receive one target.

    This function controls both:
        1. Insertion frequency
        2. Dataset-level insertion position

    Frequency is defined as the number of DISTINCT selected
    host documents.

    A given target can appear at most once in a selected host.

    Different targets may share the same host document.

    Returns
    -------
    list[int]:
        Sorted output indices of the chosen host documents.

    Why return sorted indices?
    --------------------------
    The modified dataset should preserve the original
    document sequence, independently of random selection.
    """

    mode = selection["mode"]
    total = len(hosts)

    # Map original source IDs to their indices within
    # the loaded experimental subset.
    #
    # Example:
    # hosts source IDs = [2, 5, 10]
    #
    # lookup:
    #     2  -> 0
    #     5  -> 1
    #     10 -> 2
    lookup = {
        host["source_id"]: i
        for i, host in enumerate(hosts)
    }

    # --------------------------------------------------------
    # MODE A: EXPLICIT SOURCE DOCUMENT IDS
    # --------------------------------------------------------

    if mode == "explicit":

        ids = selection["source_ids"]

        # Validate that every source ID is a valid integer.
        for x in ids:
            integer(x, "selection source ID")

        # Repeated IDs would violate distinct-host frequency.
        if len(set(ids)) != len(ids):
            raise ValueError(
                "Duplicate explicit source IDs"
            )

        # An explicitly selected host must have been loaded.
        #
        # We never silently substitute another document.
        if any(x not in lookup for x in ids):
            raise ValueError(
                "Explicit selection includes unloaded "
                "Pile source IDs"
            )

        chosen = [lookup[x] for x in ids]

    else:

        # Other modes specify frequency numerically.
        frequency = integer(
            selection["frequency"],
            "frequency"
        )

        # ----------------------------------------------------
        # MODE B: RANDOM HOSTS FROM DATASET REGION
        # ----------------------------------------------------

        if mode == "random":

            candidates = stage_positions(
                selection,
                total
            )

        # ----------------------------------------------------
        # MODE C: RANDOM HOSTS WITHIN AN INDEX RANGE
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
        # MODE D: EVERY NTH DOCUMENT
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

        # ----------------------------------------------------
        # EXCLUDE PREEXISTING TARGET OCCURRENCES
        # ----------------------------------------------------
        #
        # An important experimental consideration:
        #
        # If a host already contains the target, inserting
        # another copy would make its total occurrence count
        # different from an otherwise comparable host.
        #
        # Therefore, hosts already containing this exact
        # target string are excluded.
        #
        # This checks exact substring presence, not semantic
        # similarity or tokenizer-level equivalence.
        candidates = [
            i for i in candidates
            if span not in hosts[i]["original_text"]
        ]

        # Ensure requested frequency is feasible.
        if frequency > len(candidates):
            raise ValueError(
                f"Frequency {frequency} exceeds "
                f"{len(candidates)} eligible hosts"
            )

        if mode in ("random", "range"):

            # random.sample chooses without replacement.
            #
            # Thus every selected host is distinct.
            chosen = rng.sample(
                candidates,
                frequency
            )

        else:

            # every_n is deterministic.
            #
            # Select the first requested number of eligible
            # candidates in the chosen step pattern.
            chosen = candidates[:frequency]

    # --------------------------------------------------------
    # FINAL HOST VALIDATION
    # --------------------------------------------------------
    #
    # Explicit selection must also reject documents
    # already containing the target.
    #
    # This prevents a configuration from silently producing
    # unintended repetition counts.
    for i in chosen:

        if span in hosts[i]["original_text"]:
            raise ValueError(
                "Target already appears in original "
                f"host {hosts[i]['source_id']}"
            )

    return sorted(chosen)


# ============================================================
# 8. TOKENIZER-BASED INSERTION BOUNDARIES
# ============================================================

class TokenOffsets:
    """
    Convert token-level insertion positions into character
    offsets of the original document.

    Why is this necessary?
    ----------------------
    Python string insertion operates on character indices.

    However, experiments may require placement after a
    certain number or percentage of tokens.

    We therefore:
        1. Tokenize the original text.
        2. Retrieve its character offset mapping.
        3. Identify valid token boundaries.
        4. Convert the requested token boundary into a
           character insertion offset.

    IMPORTANT LIMITATION
    --------------------
    Inserting text may alter tokenization near the boundary.

    Therefore, an insertion at original token boundary 25
    does not necessarily remain at exactly token 25 when
    the modified document is retokenized.

    Offset mappings can also overlap for some tokenizers
    or Unicode sequences.

    Ambiguous overlapping boundaries are rejected.
    """

    def __init__(self, cfg):

        self.cfg = cfg

        # Tokenizer is initialized only on first use.
        self.tokenizer = None

    def boundaries(self, text):
        """
        Return character offsets representing token
        boundaries in the original text.

        If the original text contains N tokens,
        the returned list has N+1 entries.

        Boundary 0:
            Start of document.

        Boundary N:
            End of document.

        Interior boundaries:
            Character offsets between neighboring tokens.

        None:
            A boundary whose tokenizer offsets overlap
            and cannot safely be represented here.
        """

        # ----------------------------------------------------
        # LAZY TOKENIZER INITIALIZATION
        # ----------------------------------------------------

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

            # Slow tokenizers may not expose offset mappings.
            if not self.tokenizer.is_fast:
                raise ValueError(
                    "Tokenizer must be fast and support "
                    "offset_mapping"
                )

        # ----------------------------------------------------
        # TOKENIZE THE ORIGINAL DOCUMENT
        # ----------------------------------------------------
        #
        # add_special_tokens=False:
        #   Ignore tokenizer-specific BOS/EOS additions.
        #
        # return_offsets_mapping=True:
        #   Return character start/end for each token.
        #
        # truncation=False:
        #   Prevent silent truncation of the source document.
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False
        )

        offsets = encoded["offset_mapping"]

        # Boundary 0 corresponds to the document beginning.
        result = [0]

        # ----------------------------------------------------
        # COMPUTE INTERIOR TOKEN BOUNDARIES
        # ----------------------------------------------------

        for k in range(1, len(offsets)):

            # End of preceding token.
            prev_end = offsets[k - 1][1]

            # Beginning of following token.
            next_start = offsets[k][0]

            if prev_end > next_start:

                # Overlapping offsets indicate that this
                # boundary cannot be represented cleanly.
                result.append(None)

            else:

                # Insert immediately after the preceding
                # token's mapped character span.
                result.append(prev_end)

        # Final boundary is the end of the original string.
        result.append(len(text))

        return result


# ============================================================
# 9. RESOLVE POSITION WITHIN ONE DOCUMENT
# ============================================================

def position_offset(
    text,
    specification,
    rng,
    token_helper
):
    """
    Translate a position specification into a character
    insertion offset.

    This is the central function for within-document
    position control.

    All positions refer to the ORIGINAL host document.

    Parameters
    ----------
    text:
        Original document text.

    specification:
        Dictionary specifying insertion mode and parameters.

    rng:
        Random generator used for random positions.

    token_helper:
        TokenOffsets instance for token-based placement.

    Returns
    -------
    int:
        Character insertion offset in the original text.

    Examples
    --------
    Document = "abcdefghij"

    beginning:
        offset 0

    middle:
        offset 5

    end:
        offset 10

    character_offset=3:
        offset 3

    percentage=0.3:
        approximately offset 3
    """

    mode = specification["mode"]
    length = len(text)

    # --------------------------------------------------------
    # BEGINNING OF DOCUMENT
    # --------------------------------------------------------

    if mode == "beginning":
        return 0

    # --------------------------------------------------------
    # MIDDLE OF DOCUMENT
    # --------------------------------------------------------
    #
    # Character-based midpoint.
    # For odd lengths, integer division rounds downward.
    if mode == "middle":
        return length // 2

    # --------------------------------------------------------
    # END OF DOCUMENT
    # --------------------------------------------------------

    if mode == "end":
        return length

    # --------------------------------------------------------
    # RANDOM CHARACTER BOUNDARY
    # --------------------------------------------------------
    #
    # randint is inclusive at both endpoints.
    #
    # Possible positions:
    #     0, 1, 2, ..., length
    #
    # Thus beginning and end are also possible.
    if mode == "random_character":
        return rng.randint(0, length)

    # --------------------------------------------------------
    # PERCENTAGE OF ORIGINAL CHARACTER LENGTH
    # --------------------------------------------------------

    if mode == "percentage":

        value = fraction(
            specification["value"],
            "percentage"
        )

        # Python round() is used to choose an integer offset.
        #
        # This may split a word because percentage placement
        # does not enforce whitespace or token boundaries.
        return round(length * value)

    # --------------------------------------------------------
    # EXACT ORIGINAL CHARACTER OFFSET
    # --------------------------------------------------------

    if mode == "character_offset":

        offset = integer(
            specification["value"],
            "character_offset"
        )

        # Reject attempts to insert beyond the document end.
        if offset > length:
            raise ValueError(
                f"Character offset {offset} exceeds "
                f"document length {length}"
            )

        return offset

    # --------------------------------------------------------
    # TOKEN-BASED POSITION MODES
    # --------------------------------------------------------

    if mode in (
        "token_offset",
        "random_token",
        "token_percentage"
    ):

        # Generate the boundary mapping for this document.
        boundaries = token_helper.boundaries(text)

        # N tokens produce N+1 possible boundary entries.
        token_count = len(boundaries) - 1

        if mode == "token_offset":

            # The requested boundary index is explicit.
            k = integer(
                specification["value"],
                "token_offset"
            )

        elif mode == "token_percentage":

            # Convert a fraction of the original token count
            # into a token boundary index.
            value = fraction(
                specification["value"],
                "token_percentage"
            )

            k = round(token_count * value)

        else:

            # Randomly choose among valid token boundaries.
            #
            # Boundaries marked None represent overlapping
            # token offsets and are excluded.
            valid = [
                k
                for k, value in enumerate(boundaries)
                if value is not None
            ]

            k = rng.choice(valid)

        # Validate the token index.
        if k > token_count:
            raise ValueError(
                f"Token offset {k} exceeds original "
                f"token count {token_count}"
            )

        # Reject ambiguous tokenizer boundaries.
        if boundaries[k] is None:
            raise ValueError(
                f"Token boundary {k} overlaps in tokenizer "
                "offset mapping; choose another boundary"
            )

        # The actual string insertion always uses a
        # character offset, even in token-based modes.
        return boundaries[k]

    raise ValueError(
        f"Unknown position mode: {mode}"
    )


# ============================================================
# 10. APPLY CONTROLLED INSERTIONS
# ============================================================

def apply_insertions(
    cfg,
    hosts,
    resolved_spans=None
):
    """
    Main controlled insertion engine.

    The implementation has two conceptual phases:

    PHASE A: PLAN INSERTIONS
    ------------------------
    For each target:
        1. Resolve target text.
        2. Select host documents.
        3. Determine each host's insertion offset.
        4. Store the insertion plan.

    PHASE B: APPLY INSERTIONS
    -------------------------
    For each original host document:
        1. Collect all planned insertions.
        2. Sort them by original character offset.
        3. Construct modified text.
        4. Record inserted text locations.
        5. Verify original text preservation.

    Finally, verify insertion frequency for every target.

    WHY PLAN BEFORE APPLYING?
    -------------------------
    Several targets may be inserted into the same document.

    If we inserted each target immediately, earlier
    insertions would shift the character offsets of later
    insertions.

    By planning against the ORIGINAL text, every position
    remains independent of previously inserted targets.

    Returns
    -------
    output_documents:
        Complete selected dataset with modified text.

    target_info:
        Metadata summarizing each target and its hosts.

    insertion_records:
        Detailed records describing individual insertions.
    """

    # Use a dedicated deterministic random generator.
    rng = random.Random(cfg["seed"])

    if not hosts:
        raise ValueError("No hosts")

    # --------------------------------------------------------
    # VERIFY ORIGINAL HOST ORDER
    # --------------------------------------------------------
    #
    # The loader should already preserve source order.
    # This is an additional defensive validation.
    source_ids = [
        host["source_id"]
        for host in hosts
    ]

    if source_ids != sorted(set(source_ids)):
        raise ValueError(
            "Hosts must have distinct source IDs "
            "in original Pile order"
        )

    # Retain original texts for all insertion calculations.
    #
    # These strings must not be mutated during planning.
    originals = [
        host["original_text"]
        for host in hosts
    ]

    token_helper = TokenOffsets(cfg)

    # planned maps:
    #
    #     output_document_index -> list of insertions
    #
    # Example:
    #
    # planned[5] = [
    #     {"target_id": "target_1", "offset": 30, ...},
    #     {"target_id": "target_2", "offset": 100, ...}
    # ]
    planned = defaultdict(list)

    target_info = []

    # Prevent duplicate logical target identifiers.
    seen_ids = set()

    # ========================================================
    # PHASE A: PLAN ALL INSERTIONS
    # ========================================================

    for target_index, target in enumerate(cfg["targets"]):

        target_id = target["id"]

        if target_id in seen_ids:
            raise ValueError(
                f"Duplicate target ID: {target_id}"
            )

        seen_ids.add(target_id)

        # ----------------------------------------------------
        # STEP A1: RESOLVE TARGET SPAN
        # ----------------------------------------------------
        #
        # resolved_spans optionally lets another part of
        # the project supply already-loaded target strings.
        #
        # Otherwise, load the target using the configuration.
        if (
            resolved_spans
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
        # STEP A2: CHOOSE HOST DOCUMENTS
        # ----------------------------------------------------
        #
        # Host selection controls dataset-level placement
        # and repetition frequency.
        chosen = choose_hosts(
            target["selection"],
            hosts,
            span,
            rng
        )

        # ----------------------------------------------------
        # STEP A3: DETERMINE POSITION FOR EACH SELECTED HOST
        # ----------------------------------------------------

        for i in chosen:

            host = hosts[i]

            # Look for a document-specific position override.
            #
            # Priority:
            #   1. per_document_positions[source_id]
            #   2. default target["position"]
            #
            # Thus individual hosts can have different
            # insertion positions even for the same target.
            position_spec = target.get(
                "per_document_positions",
                {}
            ).get(
                str(host["source_id"]),
                target["position"]
            )

            # Resolve the requested position using ORIGINAL
            # text rather than previously modified text.
            offset = position_offset(
                originals[i],
                position_spec,
                rng,
                token_helper
            )

            # Store the planned insertion.
            #
            # target_order is important when two targets
            # share the same character insertion offset.
            #
            # It provides a deterministic tie-breaking order
            # matching the order of targets in CONFIG.
            planned[i].append({

                "target_id": target_id,

                "target_order": target_index,

                "span": span,

                "offset": offset,

                "position": position_spec
            })

        # ----------------------------------------------------
        # STEP A4: RECORD TARGET-LEVEL METADATA
        # ----------------------------------------------------
        #
        # This summarizes the target's actual selected hosts,
        # frequency, source configuration, and exact text.
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
    # PHASE B: BUILD THE MODIFIED DOCUMENTS
    # ========================================================

    output_documents = []

    insertion_records = []

    # Process hosts in original source order.
    #
    # Every host produces exactly one output document,
    # whether it receives zero, one, or multiple insertions.
    for i, host in enumerate(hosts):

        original_text = originals[i]

        # ----------------------------------------------------
        # SORT ALL INSERTIONS FOR THIS DOCUMENT
        # ----------------------------------------------------
        #
        # Primary key:
        #   Original character offset.
        #
        # Secondary key:
        #   Target order in CONFIG.
        #
        # This ensures deterministic handling of multiple
        # targets that share an insertion position.
        placements = sorted(
            planned[i],
            key=lambda p: (
                p["offset"],
                p["target_order"]
            )
        )

        # Instead of repeatedly modifying a Python string,
        # accumulate text fragments and join them once.
        #
        # This also makes offset tracking straightforward.
        pieces = []

        # cursor tracks how much ORIGINAL text was copied.
        cursor = 0

        # output_length tracks the constructed string length.
        #
        # It lets us record exact target locations in the
        # final modified document.
        output_length = 0

        # Records for insertions into this particular host.
        own_records = []

        # ----------------------------------------------------
        # INSERT EACH PLANNED TARGET
        # ----------------------------------------------------

        for placement in placements:

            offset = placement["offset"]

            # ------------------------------------------------
            # B1. COPY ORIGINAL TEXT BEFORE INSERTION
            # ------------------------------------------------
            #
            # Copy only original text between the previous
            # insertion point and the current insertion point.
            #
            # When two targets share an offset, this segment
            # is empty for the second insertion.
            segment = original_text[cursor:offset]

            pieces.append(segment)

            output_length += len(segment)

            # ------------------------------------------------
            # B2. ADD TEXT SEPARATORS
            # ------------------------------------------------
            #
            # Two newlines are used to visually separate
            # the inserted span from surrounding text.
            #
            # At offset 0:
            #   No leading separator is needed.
            #
            # At original document end:
            #   No trailing separator is needed.
            #
            # IMPORTANT:
            # These separators are ADDITIONAL text.
            # They are recorded and removed during
            # original-text reconstruction validation.
            #
            # They can affect the model's tokenization
            # and experimental context.
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
            # B3. CALCULATE MODIFIED-TEXT TARGET OFFSETS
            # ------------------------------------------------
            #
            # The target starts AFTER any leading separator.
            #
            # These offsets refer to the FINAL modified text,
            # not the original document.
            #
            # Python slicing uses exclusive end offsets.
            span_start = (
                output_length
                + len(left_separator)
            )

            span_end = (
                span_start
                + len(placement["span"])
            )

            # ------------------------------------------------
            # B4. APPEND THE INSERTED SPAN
            # ------------------------------------------------
            #
            # The insertion consists of:
            #
            #   optional leading separator
            #   target text
            #   optional trailing separator
            #
            # No original text is deleted here.
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
            # B5. RECORD EXACT INSERTION DETAILS
            # ------------------------------------------------
            #
            # This metadata enables:
            #   - Experimental auditing
            #   - Frequency verification
            #   - Position analysis
            #   - Exact target extraction
            #   - Reconstruction validation
            #
            # original_character_offset:
            #   Insertion boundary in original text.
            #
            # inserted_start_char / inserted_end_char:
            #   Target's character interval in modified text.
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

            # Advance the cursor to the insertion boundary
            # in the original document.
            #
            # The inserted target itself is not part of
            # the original text and does not advance cursor.
            cursor = offset

        # ----------------------------------------------------
        # B6. APPEND REMAINING ORIGINAL DOCUMENT TEXT
        # ----------------------------------------------------
        #
        # All text after the final insertion point must
        # remain in the output.
        pieces.append(original_text[cursor:])

        # Construct the complete modified document.
        modified_text = "".join(pieces)

        # ====================================================
        # PHASE C: VALIDATE ORIGINAL TEXT PRESERVATION
        # ====================================================
        #
        # Core correctness requirement:
        #
        # If all inserted spans and their separators are
        # removed, the result must exactly match the
        # original Pile document.
        #
        # This protects against accidental text deletion,
        # duplication, or offset miscalculation.
        # ====================================================

        reconstructed = modified_text

        # ----------------------------------------------------
        # REMOVE INSERTIONS IN REVERSE ORDER
        # ----------------------------------------------------
        #
        # Why reverse order?
        #
        # Removing text from the end of a string does not
        # change earlier character offsets.
        #
        # If insertions were removed from the beginning,
        # later recorded positions would become invalid.
        #
        # Reverse-order removal also handles several targets
        # inserted into the same original document.
        for record in reversed(own_records):

            start = record["inserted_start_char"]
            end = record["inserted_end_char"]

            # First verify the target really appears at
            # its recorded modified-text offsets.
            if (
                reconstructed[start:end]
                != record["inserted_span"]
            ):
                raise RuntimeError(
                    "Inserted span or offset mismatch"
                )

            # Include the leading separator in the removal.
            remove_start = (
                start
                - len(record["left_separator"])
            )

            # Include the trailing separator in the removal.
            remove_end = (
                end
                + len(record["right_separator"])
            )

            # Remove the complete insertion and join
            # the surrounding original text.
            reconstructed = (
                reconstructed[:remove_start]
                + reconstructed[remove_end:]
            )

        # ----------------------------------------------------
        # VERIFY EXACT ORIGINAL DOCUMENT RECONSTRUCTION
        # ----------------------------------------------------
        #
        # Equality is character-for-character.
        #
        # Any unintended alteration to original text causes
        # the experiment to stop with an error.
        if reconstructed != original_text:
            raise RuntimeError(
                "Original text was not preserved in "
                f"Pile document {host['source_id']}"
            )

        # ----------------------------------------------------
        # SAVE THE FINAL DOCUMENT RECORD
        # ----------------------------------------------------
        #
        # Unmodified documents have insertion_count = 0
        # and their text remains unchanged.
        #
        # Both original and modified text hashes are saved.
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
    # PHASE D: VERIFY TARGET FREQUENCY
    # ========================================================
    #
    # For every target, verify that:
    #
    # 1. The number of recorded insertions matches the
    #    planned target frequency.
    #
    # 2. Each insertion is in a distinct host document.
    #
    # This ensures that frequency means the number of
    # documents receiving the target, rather than the
    # number of repeated copies inside one document.
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
# 11. EXPERIMENT EXECUTION AND OUTPUT
# ============================================================

def main():
    """
    Execute one controlled insertion experiment.

    Workflow
    --------
    1. Validate output directory availability.
    2. Load selected Pile documents.
    3. Apply target insertions.
    4. Build final JSON dataset.
    5. Build insertion metadata.
    6. Save results.
    7. Print experimental summary.

    No model training is performed.
    """

    output_dir = Path(CONFIG["output_dir"])

    # --------------------------------------------------------
    # A. PREVENT ACCIDENTAL OVERWRITING
    # --------------------------------------------------------
    #
    # Experiments should remain independently reproducible.
    #
    # Therefore, existing output directories are rejected.
    if output_dir.exists():
        raise FileExistsError(
            "Choose a new output_dir. "
            f"Already exists: {output_dir}"
        )

    # --------------------------------------------------------
    # B. LOAD ORIGINAL HOST DOCUMENTS
    # --------------------------------------------------------

    print("Loading selected Pile documents...")

    hosts = load_hosts(CONFIG)

    print(
        f"Loaded {len(hosts)} existing Pile documents."
    )

    # --------------------------------------------------------
    # C. PERFORM CONTROLLED INSERTIONS
    # --------------------------------------------------------
    #
    # This operation performs the selection, insertion,
    # offset calculation, and correctness validation.
    print("Applying controlled insertions...")

    documents, targets, insertions = apply_insertions(
        CONFIG,
        hosts
    )

    # ========================================================
    # OUTPUT FILE 1: MODIFIED DATASET
    # ========================================================
    #
    # The result is ONE complete JSON object containing
    # all loaded documents in their original order.
    #
    # Each document has:
    #   pile_source_id
    #   output_document_index
    #   text
    #   original_sha256
    #   modified_sha256
    #   insertion_count
    #
    # Documents without insertions are retained.
    #
    # The configuration is included so the dataset can be
    # associated with the experiment that produced it.
    # ========================================================

    result = {

        "dataset": CONFIG["dataset"],

        "configuration": CONFIG,

        "document_order":
            "original_pile_stream_order_no_shuffle",

        # Dataset modification cannot guarantee actual
        # training exposure order.
        #
        # Downstream training code must explicitly preserve
        # ordering if it is part of the experiment.
        "training_time_exposure":
            "not_controlled_by_dataset_editing",

        "total_documents": len(documents),

        "total_insertions": len(insertions),

        "documents": documents
    }

    # ========================================================
    # OUTPUT FILE 2: INSERTION METADATA
    # ========================================================
    #
    # This file describes the intervention rather than
    # storing the complete modified dataset.
    #
    # It is useful for later memorization analysis.
    #
    # Example research questions:
    #
    # - Which documents received target_1?
    # - What was the exact inserted target string?
    # - At what original offsets was it inserted?
    # - What was the insertion frequency?
    # - Did target_1 appear in early or late documents?
    # - Where can inserted text be found in final documents?
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
    # D. CREATE OUTPUT DIRECTORY
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
    # E. SAVE COMPLETE MODIFIED DATASET
    # --------------------------------------------------------
    #
    # JSON format:
    # A single JSON object containing the document list.
    #
    # ensure_ascii=False preserves Unicode text.
    #
    # indent=2 makes the file human-readable.
    #
    # File mode "x" prevents overwriting an existing file.
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
    # F. SAVE EXPERIMENTAL INSERTION METADATA
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
    # 12. PRINT EXPERIMENT SUMMARY
    # ========================================================
    #
    # This summary helps researchers quickly verify:
    #
    #   - Number of processed Pile documents
    #   - Number of performed insertions
    #   - Locations of generated output files
    #   - Selected source IDs and insertion offsets
    #
    # Full experimental details remain in the metadata JSON.
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
# This ensures main() executes only when this script is run
# directly.
#
# If the script is imported by another Python module,
# its functions and configuration can be reused without
# automatically loading datasets or writing output files.
# ============================================================

if __name__ == "__main__":
    main()
